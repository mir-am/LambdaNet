package infer

import funcdiff.SimpleMath
import funcdiff.SimpleMath.Extensions._
import funcdiff.SimpleMath.{LabeledGraph, wrapInQuotes}
import gtype.GModule.ProjectPath
import gtype.{GTHole, GType}
import infer.IR._

import collection.mutable

/** Encodes the relationships between different type variables */
object PredicateGraph {

  /**
    * @param path the path of this module related to the project root
    * @param predicates all the predicates ([[TyVarPredicate]]) generated by this module
    * @param typeLabels all the user-annotated type annotations, resolved as [[IRType]]s
    */
  case class PredicateModule(
      path: ProjectPath,
      predicates: Vector[TyVarPredicate],
      newTypes: Map[IRType, TypeName],
      typeLabels: Map[IRTypeId, TypeLabel]
  ) {

    def display(srcOpt: Option[IRModule] = None): String = {
      val srcPart = srcOpt match {
        case Some(src) =>
          assert(src.path == path)
          s"== IR:\n${src.stmts.mkString("\n")}\n"
        case None => ""
      }
      s"""=== Module: $path ===
         |$srcPart== predicates:
         |${predicates.mkString("\n")}
       """.stripMargin
    }
  }

  /** The desired type annotations that the inference algorithm should
    * produce, used for supervision */
  sealed trait TypeLabel
  case object OutOfScope extends TypeLabel
  case class LibraryType(ty: GType) extends TypeLabel {
    override def toString: String = s"[L]$ty"
  }
  case class ProjectType(ty: IRType) extends TypeLabel {
    override def toString: String = s"[P]$ty"
  }

  sealed trait TyVarPredicate

  sealed trait TypingConstraint {
    def vars: Vector[IRType]
  }

  case class FreezeType(v: IRType, ty: GType) extends TyVarPredicate
  case class HasName(v: IRType, name: Symbol) extends TyVarPredicate
  case class IsLibraryType(v: IRType, name: Symbol) extends TyVarPredicate
  case class SubtypeRel(sub: IRType, sup: IRType)
      extends TyVarPredicate
      with TypingConstraint {
    val vars: Vector[IRType] = Vector(sub, sup)
  }
  case class AssignRel(lhs: IRType, rhs: IRType) extends TyVarPredicate
  case class UsedAsBoolean(tyVar: IRType) extends TyVarPredicate
  case class InheritanceRel(child: IRType, parent: IRType) extends TyVarPredicate
  case class DefineRel(v: IRType, expr: TypeExpr)
      extends TyVarPredicate
      with TypingConstraint {
    lazy val vars: Vector[IRType] = v +: expr.vars
  }

  def equalityRel(lhs: IRType, rhs: IRType): DefineRel = {
    DefineRel(lhs, VarTypeExpr(rhs))
  }

  sealed trait TypeExpr {
    def vars: Vector[IRType]
  }
  case class VarTypeExpr(v: IRType) extends TypeExpr {
    val vars: Vector[IRType] = Vector(v)
  }
  case class FuncTypeExpr(argTypes: List[IRType], returnType: IRType) extends TypeExpr {
    lazy val vars: Vector[IRType] = (argTypes :+ returnType).toVector
  }
  case class CallTypeExpr(f: IRType, args: List[IRType]) extends TypeExpr {
    lazy val vars: Vector[IRType] = (f +: args).toVector
  }
  case class ObjLiteralTypeExpr(fields: Map[Symbol, IRType]) extends TypeExpr {
    lazy val vars: Vector[IRType] = fields.values.toVector
  }
  case class FieldAccessTypeExpr(objType: IRType, field: Symbol) extends TypeExpr {
    lazy val vars: Vector[IRType] = Vector(objType)
  }

  def predicateCategory(p: TyVarPredicate): Symbol = p match {
    case _: FreezeType     => 'freeze
    case _: IsLibraryType  => 'isLibType
    case _: HasName        => 'hasName
    case _: SubtypeRel     => 'subtype
    case _: AssignRel      => 'assign
    case _: UsedAsBoolean  => 'usedAsBool
    case _: InheritanceRel => 'inheritance
    case DefineRel(_, et) =>
      et match {
        case _: VarTypeExpr         => Symbol("define-var")
        case _: FuncTypeExpr        => Symbol("define-func")
        case _: CallTypeExpr        => Symbol("define-call")
        case _: ObjLiteralTypeExpr  => Symbol("define-object")
        case _: FieldAccessTypeExpr => Symbol("define-access")
      }
  }

  def displayPredicateGraph(
      correctNodes: Seq[IRType],
      wrongNodes: Seq[IRType],
      predicates: Seq[TyVarPredicate],
      typeHoleMap: Map[IRTypeId, GTHole]
  ): LabeledGraph = {
    def typeInfo(t: IR.IRType): String = {
      val holeInfo = typeHoleMap.get(t.id).map(h => s";Hole: ${h.id}").getOrElse("")
      wrapInQuotes(t.toString.replace("\uD835\uDCAF", "") + holeInfo)
    }

    var nodeId = 0
    def newNode(): Int = {
      nodeId -= 1
      nodeId
    }

    val graph = new SimpleMath.LabeledGraph()

    def newPredicate(
        shortName: String,
        fullName: String,
        connections: Seq[(Int, String)]
    ): Unit = {
      val n = newNode()
      graph.addNode(n, shortName, wrapInQuotes(fullName), "Blue")
      connections.foreach {
        case (id, label) =>
          graph.addEdge(n, id, wrapInQuotes(label))
      }
    }

    correctNodes.foreach(n => {
      graph.addNode(n.id, n.id.toString, typeInfo(n), "Green")
    })
    wrongNodes.foreach(n => {
      graph.addNode(n.id, n.id.toString, typeInfo(n), "Red")
    })

    predicates.foreach {
      case FreezeType(v, ty) =>
        newPredicate(s"=$ty", s"Freeze to $ty", Seq(v.id -> ""))
      case IsLibraryType(v, name) =>
        newPredicate(s"[${name.name}]", s"Is library type: $name", Seq(v.id -> ""))
      case HasName(v, name) =>
        newPredicate(s"{${name.name}}", s"Has name $name", Seq(v.id -> ""))
      case SubtypeRel(sub, sup) =>
        newPredicate("<:", "Subtype", Seq(sub.id -> "sub", sup.id -> "sup"))
      case AssignRel(lhs, rhs) =>
        newPredicate(":=", "Assign", Seq(lhs.id -> "lhs", rhs.id -> "rhs"))
      case UsedAsBoolean(tyVar) =>
        newPredicate("~bool", "Use as bool", Seq(tyVar.id -> ""))
      case InheritanceRel(child, parent) =>
        newPredicate(
          "extends",
          "Extends",
          Seq(child.id -> "child", parent.id -> "parent")
        )
      case DefineRel(v, expr) =>
        val (short, long, connections) = expr match {
          case VarTypeExpr(v1) =>
            ("Var", "VarTypeExpr", Vector(v1.id -> "var"))
          case FuncTypeExpr(argTypes, returnType) =>
            val conn = argTypes.zipWithIndex.map { case (a, i) => a.id -> s"arg$i" } :+ (returnType.id -> "return")
            ("Func", "FuncTypeExpr", conn)
          case CallTypeExpr(f, args) =>
            val conn = args.zipWithIndex.map { case (a, i) => a.id -> s"arg$i" } :+ (f.id -> "f")
            ("Call", "CallTypeExpr", conn)
          case ObjLiteralTypeExpr(fields) =>
            val conn = fields.toList.map { case (label, t) => t.id -> label.toString() }
            ("Obj", "ObjLiteralTypeExpr", conn)
          case FieldAccessTypeExpr(objType, field) =>
            (s"_.${field.name}", "FieldAccessTypeExpr", Seq(objType.id -> ""))
        }
        newPredicate(short, long, connections :+ (v.id -> "="))
    }

    graph
  }

  val returnVar: Var = namedVar(gtype.GStmt.returnSymbol)

}
