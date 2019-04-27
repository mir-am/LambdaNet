package infer

import funcdiff.IS
import gtype.GType
import infer.IR.{IRType, IRTypeId}
import infer.PredicateGraph._
import org.platanios.tensorflow.api._

import collection.mutable
import scala.language.implicitConversions
import infer.GraphEmbeddingTF._
import funcdiff.SimpleMath.Extensions._

object GraphEmbeddingTF {
  type D = Float
  type CompNode = Output[D]

  val unknownTypeSymbol = 'UNKNOWN

  case class EmbeddingCtx(
    idTypeMap: Map[IRTypeId, IRType],
    libraryTypeMap: Symbol => CompNode,
    labelMap: Symbol => CompNode,
    fieldKnowledge: Map[Symbol, (CompNode, CompNode)],
    varKnowledge: Map[Symbol, CompNode]
  )

  case class Embedding(nodeMap: Map[IR.IRTypeId, CompNode], stat: EmbeddingStat)

  case class EmbeddingStat(trueEmbeddingLengths: Vector[CompNode])

}

case class EmbeddingAPI(dimMessage: Int) {
  def randomVec(name: String): CompNode =
    tf.variable[D](
        name,
        Shape(1, dimMessage),
        initializer = tf.RandomNormalInitializer(standardDeviation = 0.01f)
      )
      .value

  implicit def symbol2String(s: Symbol): String = s.name
  implicit class SymbolExt(s: Symbol) {
    def /(s1: Symbol): String = s.name + "_" + s1.name
  }
}

case class GraphEmbeddingTF(
  predicates: Seq[TyVarPredicate],
  ctx: EmbeddingCtx,
  dimMessage: Int
) {

  import ctx._

  require(dimMessage % 2 == 0, "dimMessage should be even")

  implicit var mode: tf.learn.Mode = tf.learn.TRAINING
  private val api = EmbeddingAPI(dimMessage)
  import api._

  val nodeInitVec: CompNode = randomVec("nodeInitVec")
  val knowledgeMissing: CompNode = randomVec("knowledgeMissing")

  /** Each message consists of a key-value pair */
  type Message = (CompNode, CompNode)

  def linear(name: String, outDim: Int)(input: CompNode): CompNode = {
    val layer = tf.learn
      .Linear[D](
        tf.currentNameScope + name,
        outDim,
        useBias = true,
        weightsInitializer = tf.VarianceScalingInitializer()
      )
    layer.forward(input)
  }

  def singleLayer(name: String, input: CompNode): CompNode =
    tf.createWith(nameScope = name) {
      tf.relu(linear("linear", dimMessage)(input))
    }

  def messageModel(name: String, vec: CompNode): Message =
    tf.createWith(nameScope = name) {
      singleLayer("header", vec) -> singleLayer("content", vec)
    }

  def binaryMessage(name: String, v1: CompNode, v2: CompNode): Message =
    tf.createWith(nameScope = name) {
      val together = tf.concatenate(Seq(v1, v2), axis = 1)
      singleLayer("header", together) -> singleLayer("content", together)
    }

  def fieldAccessMessage(
    name: String,
    objEmbed: CompNode,
    fieldLabel: Symbol
  ): Message = tf.createWith(nameScope = name) {
    val input = tf.concatenate(Seq(objEmbed, labelMap(fieldLabel)), axis = 1)
    messageModel("message", singleLayer("compress", input))
  }

  def argAccessMessage(name: String, fEmbed: CompNode, argId: Int): Message =
    tf.createWith(nameScope = name) {
      val input = tf.concatenate(Seq(fEmbed, positionalEncoding(argId)), axis = 1)
      messageModel("message", singleLayer("compress", input))
    }

  private val posEncodingCache = collection.mutable.HashMap[Int, CompNode]()
  def positionalEncoding(pos: Int): CompNode = {
    assert(pos >= -1)
    posEncodingCache.getOrElseUpdate(
      pos,
      tf.createWith(nameScope = "position/") {
        if (pos == -1)
          randomVec("head")
        else {
          val phases = (0 until dimMessage / 2).map { dim =>
            pos / math.pow(1000, 2.0 * dim / dimMessage)
          }
          val ts = Tensor((phases.map(math.sin) ++ phases.map(math.cos)).map(_.toFloat))
            .reshape(Shape(1, dimMessage))

          tf.constant(ts, name = s"POS_$pos")
        }
      }
    )
  }

  /** performs weighted-sum over ys using dot-product attention */
  def attentionLayer(
    name: String,
    transformKey: Boolean = false,
    transformValue: Boolean = false
  )(xKey: CompNode, ys: IS[(CompNode, CompNode)]): CompNode =
    tf.createWith(nameScope = name) {
      require(ys.nonEmpty)
      val keyDim = xKey.shape(1)
      val valueDim = ys.head._2.shape(1)
      val sqrtN = math.sqrt(keyDim).toFloat
      val weightLogits = {
        val originalKeys = tf.concatenate(ys.map(_._1), axis = 0)
        val keys =
          if (transformKey)
            linear("keyTransform2", keyDim)(originalKeys)
          else originalKeys
        tf.matmul(keys, xKey.transpose()).transpose()
      }
      val aWeights = tf.softmax(tf.divide[D](weightLogits, sqrtN))

      assert(aWeights.shape(0) == 1)
      val yOrigin = tf.concatenate(ys.map(_._2), axis = 0)
      val yMat =
        if (transformValue) tf.relu(linear("valueTransform", valueDim)(yOrigin))
        else yOrigin
      tf.matmul(aWeights, yMat)
    }

  /**
    * @return A prediction distribution matrix of shape (# of places) * (# of candidates)
    */
  def encodeAndDecode(
    iterations: Int
//    decodingCtx: DecodingCtx,
//    placesToDecode: IS[IRTypeId]
  ) = {
    val stat = EmbeddingStat(
      idTypeMap.mapValuesNow(_ => tf.constant(Tensor(1f))).values.toVector
    )
    val init = Embedding(idTypeMap.mapValuesNow(_ => nodeInitVec), stat)
    val embeddings = IS.iterate(init, iterations)(iterate)

//    val result = DebugTime.logTime('decodingTime) {
//      decode(decodingCtx, placesToDecode, embeddings.last)
//    }

    embeddings
  }

  def iterate(
    embedding: Embedding
  ): Embedding = {
    import embedding._

    val messages =
      ctx.idTypeMap.keys
        .map(id => id -> mutable.ListBuffer[Message]())
        .toMap

    type ObjType = IRType
    val fieldDefs = mutable.HashMap[Symbol, IS[(ObjType, IRType)]]()
    val fieldUsages = mutable.HashMap[Symbol, IS[(ObjType, IRType)]]()
    predicates.collect {
      case DefineRel(v, ObjLiteralTypeExpr(fields)) =>
        fields.foreach {
          case (l, t) =>
            fieldDefs(l) = fieldDefs.getOrElse(l, IS()) :+ (v, t)
        }
      case DefineRel(v, FieldAccessTypeExpr(objType, l)) =>
        fieldUsages(l) = fieldUsages.getOrElse(l, IS()) :+ (objType, v)
    }

    /* for each kind of predicate, generate one or more messages */
    def sendPredicateMessages(predicate: TyVarPredicate): Unit = predicate match {
      case EqualityRel(v1, v2) =>
        messages(v1.id) += binaryMessage('SubtypeRel, nodeMap(v1.id), nodeMap(v2.id))
        messages(v2.id) += binaryMessage('SubtypeRel, nodeMap(v2.id), nodeMap(v1.id))
      case FreezeType(v, ty) =>
        messages(v.id) += messageModel('FreezeType, encodeGType(ty))
      case IsLibraryType(v, name) =>
        val knowledge = varKnowledge.getOrElse(name, knowledgeMissing)
        messages(v.id) += messageModel('IsLibrary, knowledge)
      case HasName(v, name) =>
      //        messages(v.id) += messageModel('HasName, labelMap(name)) //todo: properly handle name info
      case SubtypeRel(sub, sup) =>
        messages(sub.id) += binaryMessage('SubtypeRel, nodeMap(sub.id), nodeMap(sup.id))
        messages(sup.id) += binaryMessage('SubtypeRel, nodeMap(sup.id), nodeMap(sub.id))
      case AssignRel(lhs, rhs) =>
        messages(lhs.id) += binaryMessage('AssignRel, nodeMap(lhs.id), nodeMap(rhs.id))
        messages(rhs.id) += binaryMessage('AssignRel, nodeMap(rhs.id), nodeMap(lhs.id))
      case UsedAsBoolean(tyVar) =>
        messages(tyVar.id) += messageModel('UsedAsBoolean, nodeMap(tyVar.id))
      case InheritanceRel(child, parent) =>
        messages(child.id) += binaryMessage(
          'DeclaredAsSubtype,
          nodeMap(child.id),
          nodeMap(parent.id)
        )
        messages(parent.id) += binaryMessage(
          'DeclaredAsSupertype,
          nodeMap(child.id),
          nodeMap(parent.id)
        )
      case DefineRel(v, expr) =>
        expr match {
          case FuncTypeExpr(argTypes, returnType) =>
            val ids = (returnType +: argTypes.toIndexedSeq).map(_.id)
            ids.zipWithIndex.foreach {
              case (tId, i) =>
                val argId = i - 1
                messages(v.id) += argAccessMessage(
                  'FuncTypeExpr / 'toF,
                  nodeMap(tId),
                  argId
                )
                messages(tId) += argAccessMessage(
                  'FuncTypeExpr / 'toArg,
                  nodeMap(v.id),
                  argId
                )
            }
          case CallTypeExpr(f, args) =>
            val fVec = nodeMap(f.id)
            val fKey = singleLayer('CallTypeExpr / 'fKey, fVec)
            val fPair = messageModel('CallTypeExpr / 'fPair, fVec)

            val argPairs = args.toIndexedSeq.zipWithIndex.map {
              case (argT, argId) =>
                argAccessMessage(
                  'CallTypeExpr / 'embedArg,
                  nodeMap(argT.id),
                  argId
                )
            }

            val fEmbed = attentionLayer('CallTypeExpr / 'fEmbed)(
              fKey,
              fPair +: argPairs
            )
            messages(v.id) += messageModel('CallTypeExpr / 'toV, fEmbed)
            args.zipWithIndex.foreach {
              case (arg, argId) =>
                messages(arg.id) += argAccessMessage(
                  'CallTypeExpr / 'toArg,
                  fEmbed,
                  argId
                )
            }
          case ObjLiteralTypeExpr(fields) =>
            fields.foreach {
              case (label, tv) =>
                messages(v.id) += fieldAccessMessage(
                  'ObjLiteralTypeExpr / 'toV,
                  nodeMap(tv.id),
                  label
                )
                messages(tv.id) += fieldAccessMessage(
                  'ObjLiteralTypeExpr / 'toField,
                  nodeMap(v.id),
                  label
                )
                if (fieldUsages.contains(label)) {
                  messages(tv.id) += {
                    val att =
                      attentionLayer(
                        'ObjLiteralTypeExpr / 'fieldUsage,
                        transformKey = true
                      )(
                        nodeMap(v.id),
                        fieldUsages(label).map {
                          case (k, n) => nodeMap(k.id) -> nodeMap(n.id)
                        }
                      )
                    messageModel('ObjLiteralTypeExpr / 'usageMessage, att)
                  }
                }
            }
          case FieldAccessTypeExpr(objType, label) =>
            messages(v.id) += fieldAccessMessage(
              'FieldAccess / 'toV,
              nodeMap(objType.id),
              label
            )
            messages(objType.id) += fieldAccessMessage(
              'FieldAccess / 'toObj,
              nodeMap(v.id),
              label
            )
            if (fieldDefs.contains(label)) {
              messages(v.id) += {
                val att =
                  attentionLayer('FieldAccess / 'defs, transformKey = true)(
                    nodeMap(objType.id),
                    fieldDefs(label)
                      .map { case (k, n) => nodeMap(k.id) -> nodeMap(n.id) } ++
                      fieldKnowledge.get(label).toIndexedSeq
                  )
                messageModel('FieldAccess / 'defsMessage, att)
              }
            }
        }
    }

    predicates.foreach(sendPredicateMessages)

    val outLengths = mutable.ListBuffer[CompNode]()
    val newNodeMap = messages.keys.toSeq
      .map { id =>
        val node = nodeMap(id)
        val out = attentionLayer('MessageAggregate, transformKey = true)(
          node,
          messages(id).toIndexedSeq :+ (node, node)
        )
        val outLen = tf.sqrt(tf.sum(tf.square(out)))
        outLengths += outLen
        val newEmbed = out / outLen
        //      val newEmbed = gru('MessageAggregate / 'updateGru)(nodeVec, change)
        id -> newEmbed
      }
      .seq
      .toMap
    TrainingCenter.note("iterate/After updating")
    val stat = EmbeddingStat(outLengths.toVector)
    Embedding(newNodeMap, stat)
  }

  private val gTypeEmbeddingMap = mutable.HashMap[GType, CompNode]()
  private val funcInitKey = randomVec('funcType / 'initKey)
//  private val funcInitValue = randomVec('funcType / 'initValue)
  private val objectInitKey = randomVec('objType / 'initKey)
//  private val objectInitValue = randomVec('objType / 'initValue)
  private val anyTypeEncoding = randomVec('encodeGType / 'anyType)
  private val emptyObject = randomVec('encodeGType / 'emptyObject)
  def encodeGType(ty: GType): CompNode = tf.nameScope("encodeGType/") {
    import gtype._

    def rec(ty: GType): CompNode = ty match { //todo: need better ways
      case AnyType     => anyTypeEncoding
      case TyVar(name) => libraryTypeMap(name)
      case FuncType(args, to) =>
        val messages = (to +: args).zipWithIndex.map {
          case (t, i) =>
            argAccessMessage('funcType / 'arg, rec(t), i - 1)
        }.toIndexedSeq
        attentionLayer('funcType / 'aggregate)(
          funcInitKey,
          messages
        )
      case ObjectType(fields) =>
        if (fields.isEmpty) {
          emptyObject
        } else {
          val messages = fields.toIndexedSeq.map {
            case (label, t) =>
              fieldAccessMessage('objectType / 'field, rec(t), label)
          }
          attentionLayer('objectType / 'aggregate)(
            objectInitKey,
            messages
          )
        }
    }

    gTypeEmbeddingMap.getOrElseUpdate(ty, rec(ty))
  }
}
