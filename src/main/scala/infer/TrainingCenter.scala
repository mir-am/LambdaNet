package infer

import java.util.concurrent.{ForkJoinPool, TimeoutException}

import botkop.numsca
import botkop.numsca.Tensor
import funcdiff.API._
import funcdiff.SimpleMath.BufferedTotalMap
import funcdiff.SimpleMath.Extensions._
import funcdiff._
import gtype.EventLogger.PlotConfig
import gtype._
import infer.GraphEmbedding._
import infer.IRTranslation.TranslationEnv

import scala.collection.mutable
import scala.collection.parallel.ForkJoinTaskSupport
import ammonite.ops._
import infer.IR.{IRModule, IRType, IRTypeId}
import infer.PredicateGraph.{
  LibraryType,
  OutOfScope,
  PredicateModule,
  ProjectType,
  TypeLabel
}
import infer.PredicateGraphConstruction.{LibraryContext, ParsedProject, PathMapping}

import scala.concurrent.{Await, ExecutionContextExecutorService, Future}
import scala.util.Random

/**
  * How to control the training:
  * * To stop: Inside 'running-result/control', put a file called 'stop.txt' to make the
  *   training stop and save at the beginning of the next training step.
  *
  * * To restore: Inside 'running-result/control', put a file called 'restore.txt' with
  *   the path pointing to a trainingState.serialized file inside it before the training
  *   starts to restore the training state.
  */
object TrainingCenter {

  val numOfThreads
    : Int = Runtime.getRuntime.availableProcessors().min(12) //use at most 12 cores
  val forkJoinPool = new ForkJoinPool(numOfThreads)
  val taskSupport: ForkJoinTaskSupport = new ForkJoinTaskSupport(forkJoinPool)
  val parallelCtx: ExecutionContextExecutorService =
    concurrent.ExecutionContext.fromExecutorService(forkJoinPool)

  TensorExtension.checkNaN = false // skip checking to train faster

  /** A complete representation of the current training, used to save/restore training */
  case class TrainingState(
    step: Int,
    dimMessage: Int,
    layerFactory: LayerFactory,
    optimizer: Optimizer
  ) {
    def saveToFile(file: Path): Unit = {
      val toSave =
        (step, dimMessage, layerFactory.paramCollection.toSerializable, optimizer)
      SimpleMath.saveObjectToFile(file.toIO)(toSave)
    }
  }

  object TrainingState {
    def fromFile(file: Path): TrainingState = {
      val (step, dimMessage, data, optimizer) = SimpleMath
        .readObjectFromFile[(Int, Int, ParamCollection.SerializableFormat, Optimizer)](
          file.toIO
        )
      val factory = LayerFactory(
        SymbolPath.empty / 'TypingNet,
        ParamCollection.fromSerializable(data)
      )
      TrainingState(step, dimMessage, factory, optimizer)
    }
  }

  def main(args: Array[String]): Unit = {
    println(s"Using threads: $numOfThreads")

//    val libraryTypes = JSExamples.libraryTypes

//    val trainRoot = pwd / RelPath("data/toy")
//    val testRoot = trainRoot
    println("Start loading projects")
    val (trainParsed, testParsed) = {
      val all = TrainingProjects.parsedProjects
      (all.drop(3), all.take(3))
    }
    println("Training/testing projects loaded")

    println(s"=== Training on ${trainParsed.map(_.projectName)} ===")

    val loadFromFile: Option[Path] = TrainingControl.restoreFromFile(consumeFile = true)
    val trainingState = loadFromFile
      .map { p =>
        println("Loading training from file: " + p)
        TrainingState.fromFile(p)
      }
      .getOrElse(
        TrainingState(
          step = 0,
          layerFactory = LayerFactory(SymbolPath.empty / 'TypingNet, ParamCollection()),
          dimMessage = 64,
          optimizer = Optimizers.Adam(learningRate = 4e-4)
        )
      )

    trainOnModules(
      trainParsed,
      testParsed,
      trainingState
    )
  }

  /** Builds a graph neural network over an entire typescript project */
  //noinspection TypeAnnotation
  case class GraphNetBuilder(
    graphName: String,
    predModules: IS[PredicateModule],
    transEnv: TranslationEnv,
    libraryVars: Vector[Symbol],
    libraryFields: Vector[Symbol],
    libraryTypes: Vector[GType],
    factory: LayerFactory,
    dimMessage: Int = 64
  ) {

    val typeLabels = predModules.flatMap(m => m.typeLabels)

    val predicates = predModules.flatMap(m => m.predicates) ++ PredicateGraphConstruction
      .encodeUnaryPredicates(transEnv.idTypeMap.values)
    val newTypes = predModules.flatMap(m => m.newTypes.keys).toSet

    def predicateCategoryNumbers: Map[Symbol, Int] = {
      predicates.groupBy(PredicateGraph.predicateCategory).mapValuesNow { _.length }
    }

    val decodingCtx = DecodingCtx(
      Vector(TyVar(unknownTypeSymbol)) ++ libraryTypes,
      newTypes.toVector
    )

    import factory._

    def randomVar(path: SymbolPath): CompNode =
      getVar(path)(numsca.randn(1, dimMessage) * 0.01)

    import GraphEmbedding._

    def encodeDecode(): (CompNode, IS[Embedding]) = {
      val libraryTypeMap: Map[Symbol, CompNode] = {
        libraryTypes.collect {
          case TyVar(s) => //only type vars, other types would be deconstructed
            s -> randomVar('TyVar / s)
        }
      }.toMap

      val varKnowledge = (libraryVars ++ libraryTypes
        .collect { case TyVar(s) => s }).map { s =>
        s -> randomVar('libVarKnowledge / s)
      }.toMap

      val fieldKnowledge =
        libraryFields.map { k =>
          k -> (randomVar('fieldKey / k), randomVar('fieldValue / k))
        }.toMap

      val unknownTypeVec = randomVar('TyVar / GraphEmbedding.unknownTypeSymbol)

      val extendedTypeMap = BufferedTotalMap(libraryTypeMap.get) { _ =>
        unknownTypeVec
      }

      val labelEncoding = BufferedTotalMap((_: Symbol) => None) { _ =>
        const(TensorExtension.randomUnitVec(dimMessage)) ~>
          linear('UnknownLabel, dimMessage) ~> relu //todo: see if this is useful
      }

      val embedCtx = EmbeddingCtx(
        transEnv.idTypeMap.toMap,
        extendedTypeMap,
        predicates,
        labelEncoding,
        fieldKnowledge,
        varKnowledge
      )

      Await.result(
        Future(
          GraphEmbedding(embedCtx, factory, dimMessage, Some(taskSupport))
            .encodeAndDecode(
              iterations = 10,
              decodingCtx,
              typeLabels.map(_._1)
            )
        )(parallelCtx),
        Timeouts.encodeDecodeTimeout
      )
    }

  }

  def trainOnModules(
    trainingProjects: IS[ParsedProject],
    testingModules: IS[ParsedProject],
    trainingState: TrainingState
  ): Unit = {

    val (machineName, emailService) = ReportFinish.readEmailInfo()

    val TrainingState(initStep, dimMessage, factory, optimizer) = trainingState

    /** any symbols that are not defined within the project */
    val libraryFields: Vector[Symbol] = {
      var allDefined, allUsed = Set[Symbol]()
      trainingProjects.foreach { p =>
        p.irModules.foreach(m => {
          val stats = m.moduleStats
          allDefined ++= stats.fieldsDefined
          allUsed ++= stats.fieldsUsed
        })
      }
      (allUsed -- allDefined).toVector
    }
    println("libraryFields: " + libraryFields)

    val libraryTypes = {
      val totalFreq = mutable.HashMap[GType, Int]()
      trainingProjects.foreach { p =>
        p.libCtx.libraryTypeFreq.foreach {
          case (t, f) =>
            totalFreq(t) = totalFreq.getOrElse(t, 0) + f
        }
      }
      val all = totalFreq.toVector.sortBy(_._2).reverse
      println("Total number of lib types: " + all.length)
      val freqs = all.map(_._2)
      println("Total usages: " + freqs.sum)
      val numOfTypes = 100
      val ratio = freqs.take(numOfTypes).sum.toDouble / freqs.sum
      println(
        s"Take at most the first $numOfTypes types, results in %${ratio * 100} coverage."
      )
      val taken = all.take(numOfTypes).map(_._1)
      println(s"Types taken: $taken")
      taken
    }

    val libraryVars = trainingProjects.flatMap(p => p.libCtx.libraryVars.keySet).toVector

    val trainBuilders = trainingProjects.map { p =>
      GraphNetBuilder(
        p.projectName,
        p.predModules,
        p.libCtx.transEnv,
        libraryVars,
        libraryFields,
        libraryTypes,
        factory,
        dimMessage
      )
    }

    val testBuilders = testingModules.map { p =>
      GraphNetBuilder(
        p.projectName,
        p.predModules,
        p.libCtx.transEnv,
        libraryVars,
        libraryFields,
        libraryTypes,
        factory,
        dimMessage
      )
    }

    trainBuilders.foreach(builder => {
      val total = builder.predicates.length
      println(s"# of predicates: $total")
      println {
        builder.predicateCategoryNumbers.toVector
          .sortBy { case (_, n) => -n }
          .map {
            case (cat, n) => s"$cat -> %.1f".format(n.toDouble / total * 100) + "%"
          }
      }
      println("# of nodes: " + builder.transEnv.idTypeMap.size)
    })

    val eventLogger = {
      import ammonite.ops._
      new EventLogger(
        pwd / "running-result" / "log.txt",
        printToConsole = true,
        overrideMode = true,
        configs = Seq(
          //          "embedding-magnitudes" -> PlotConfig("ImageSize->Medium"),
          "embedding-changes" -> PlotConfig("ImageSize->Medium"),
//          "embedding-max-length" -> PlotConfig("ImageSize->Medium"),
          "iteration-time" -> PlotConfig(
            "ImageSize->Medium",
            """AxesLabel->{"step","ms"}"""
          ),
          "loss" -> PlotConfig("ImageSize->Large"),
          "accuracy" -> PlotConfig("ImageSize->Medium"),
          "test-accuracy" -> PlotConfig("ImageSize->Medium"),
          "test-lib-accuracy" -> PlotConfig("ImageSize->Small"),
          "test-proj-accuracy" -> PlotConfig("ImageSize->Small")
        )
      )
    }

    import TensorExtension.oneHot

    val maxTrainingSteps = 1000
    // training loop
    for (step <- initStep + 1 to maxTrainingSteps) try {
      if (TrainingControl.shouldStop(true)) {
        saveTraining(step - 1, s"stopped-step$step")
        throw new Exception("Stopped by 'stop.txt'.")
      }

      val startTime = System.currentTimeMillis()

      for (trainBuilder <- trainBuilders) {
        println(s"training on ${trainBuilder.graphName}...")

        val (logits, embeddings) = {
          note("encodeDecode")
          trainBuilder.encodeDecode()
        }

        DebugTime.logTime('loggingTime) {
          note("loggingTime")
          val diffs = embeddings
            .zip(embeddings.tail)
            .par
            .map {
              case (e1, e0) =>
                val diffMap = e1.nodeMap.elementwiseCombine(e0.nodeMap) { (x, y) =>
                  math.sqrt(numsca.sum(numsca.square(x.value - y.value)))
                }
                SimpleMath.mean(diffMap.values.toSeq)
            }
            .seq
          eventLogger.log("embedding-changes", step, Tensor(diffs: _*))

//          val maxEmbeddingLength = embeddings.map {
//            _.stat.trueEmbeddingLengths.max
//          }.max
//          eventLogger.log("embedding-max-length", step, Tensor(maxEmbeddingLength))
        }

        note("analyzeResults")
        val typeLabels = trainBuilder.typeLabels
        val decodingCtx = trainBuilder.decodingCtx
        println("max Idx: " + decodingCtx.maxIndex)

        val accuracy = analyzeResults(
          typeLabels,
          logits.value,
          trainBuilder.transEnv,
          decodingCtx,
          printResults = None
        )
        eventLogger.log("accuracy", step, Tensor(accuracy.totalAccuracy))

        println("annotated places number: " + typeLabels.length)
        val targets = typeLabels.map(p => decodingCtx.indexOfType(p._2))
        val loss = mean(
          crossEntropyOnSoftmax(logits, oneHot(targets, decodingCtx.maxIndex))
        )

        if (loss.value.squeeze() > 20) {
          println(s"Abnormally large loss: ${loss.value}")
          println("logits: ")
          println {
            logits.value
          }
        }
        eventLogger.log("loss", step, loss.value)

        note("optimization")
        DebugTime.logTime('optimization) {
          optimizer.minimize(
            loss,
            trainBuilder.factory.paramCollection.allParams,
            backPropInParallel = Some(parallelCtx -> Timeouts.optimizationTimeout)
          )
        }

        eventLogger.log(
          "iteration-time",
          step,
          Tensor(System.currentTimeMillis() - startTime)
        )
      }

      println(DebugTime.show)

      if (step % 10 == 0) {
        println("start testing...")
        val testAccs = testBuilders.map { testBuilder =>
          val (testLogits, _) = testBuilder.encodeDecode()
          val testAcc = analyzeResults(
            testBuilder.typeLabels,
            testLogits.value,
            testBuilder.transEnv,
            testBuilder.decodingCtx
          )
          testAcc
        }
        val avgAcc = SimpleMath.mean(testAccs.map(_.totalAccuracy))
        val avgLibAcc = SimpleMath.mean(testAccs.map(_.libraryTypeAccuracy))
        val avgProjAcc = SimpleMath.mean(testAccs.map(_.projectTypeAccuracy))
        eventLogger.log("test-accuracy", step, Tensor(avgAcc))
        eventLogger.log("test-lib-accuracy", step, Tensor(avgLibAcc))
        eventLogger.log("test-proj-accuracy", step, Tensor(avgProjAcc))
      }
      if (step % 50 == 0) {
        saveTraining(step, s"step$step")
      }
    } catch {
      case ex: Throwable =>
        val isTimeout = ex.isInstanceOf[TimeoutException]
        val errorName = if (isTimeout) "timeout" else "stopped"
        emailService.sendMail(emailService.userEmail)(
          s"TypingNet: $errorName on $machineName at step $step",
          s"Details:\n" + ex.getMessage
        )
        if (isTimeout && Timeouts.restartOnTimeout) {
          println("Timeout... training restarted (skip one training step)...")
        } else {
          saveTraining(step, "error-save")
          throw ex
        }
    }

    emailService.sendMail(emailService.userEmail)(
      s"TypingNet: Training finished on $machineName!",
      "Training finished!"
    )

    saveTraining(maxTrainingSteps, "finished")

    def saveTraining(step: Int, dirName: String): Unit = {
      println("save training...")
      val saveDir = pwd / "running-result" / "saved" / dirName
      if (!exists(saveDir)) {
        mkdir(saveDir)
      }
      val savePath = saveDir / "trainingState.serialized"
      TrainingState(step, dimMessage, factory, optimizer).saveToFile(savePath)
      println("Training state saved into: " + saveDir)
    }

  }

  object Timeouts {
    import concurrent.duration._

    var restartOnTimeout = true
    var optimizationTimeout = 1000.seconds
    var encodeDecodeTimeout = 400.seconds
  }

  case class AccuracyStats(
    totalAccuracy: Double,
    projectTypeAccuracy: Double,
    libraryTypeAccuracy: Double,
    outOfScopeTypeAccuracy: Double
  )

  def analyzeResults(
    annotatedPlaces: IS[(IRTypeId, TypeLabel)],
    logits: Tensor,
    transEnv: TranslationEnv,
    ctx: DecodingCtx,
    printResults: Option[Int] = Some(100)
  ): AccuracyStats = {
    type Prediction = Int
    val predictions = numsca.argmax(logits, axis = 1)
    val correct = mutable.ListBuffer[(IRTypeId, Prediction)]()
    val incorrect = mutable.ListBuffer[(IRTypeId, Prediction)]()
    var projCorrect, projIncorrect = 0
    var libCorrect, libIncorrect = 0
    var outOfScopeCorrect, outOfScopeIncorrect = 0

    for (row <- annotatedPlaces.indices) {
      val (nodeId, t) = annotatedPlaces(row)
      val expected = ctx.indexOfType(t)
      val actual = predictions(row, 0).squeeze().toInt
      if (expected == actual) {
        correct += (nodeId -> actual)
        t match {
          case _: ProjectType => projCorrect += 1
          case _: LibraryType => libCorrect += 1
          case OutOfScope     => outOfScopeCorrect += 1
        }
      } else {
        incorrect += (nodeId -> actual)
        t match {
          case _: ProjectType => projIncorrect += 1
          case _: LibraryType => libIncorrect += 1
          case OutOfScope     => outOfScopeIncorrect += 1
        }
      }
    }

    printResults.foreach { num =>
      val rand = new Random()
      rand.shuffle(correct).take(num).foreach {
        case (id, pred) =>
          val t = ctx.typeFromIndex(pred)
          val tv = transEnv.idTypeMap(id)
          println(s"[correct] \t$tv: $t")
      }
      val labelMap = annotatedPlaces.toMap
      rand.shuffle(incorrect).take(num).foreach {
        case (id, pred) =>
          val tv = transEnv.idTypeMap(id)
          val actualType = ctx.typeFromIndex(pred)
          val expected = labelMap(id)
          println(s"[incorrect] \t$tv: $actualType not match $expected")
      }
    }

    def calcAccuracy(correct: Int, incorrect: Int): Double = {
      val s = correct + incorrect
      if (s == 0) 1.0
      else correct.toDouble / s
    }

    val accuracy = correct.length.toDouble / (correct.length + incorrect.length)
    val libAccuracy = calcAccuracy(libCorrect, libIncorrect)
    val projAccuracy = calcAccuracy(projCorrect, projIncorrect)
    val outOfScopeAccuracy = calcAccuracy(outOfScopeCorrect, outOfScopeIncorrect)

    AccuracyStats(accuracy, libAccuracy, projAccuracy, outOfScopeAccuracy)
  }

  def note(msg: String): Unit = {
    val printNote = false
    if (printNote) println("note: " + msg)
  }

  /** Use text files to control the training loop (stop, restore, etc) */
  object TrainingControl {
    val stopFile: Path = pwd / "running-result" / "control" / "stop.txt"
    val restoreFile: Path = pwd / "running-result" / "control" / "restore.txt"

    def shouldStop(consumeFile: Boolean): Boolean = {
      val stop = exists(stopFile)
      if (consumeFile && stop) {
        rm(stopFile)
      }
      stop
    }

    /** If [[restoreFile]] exists, read the path from the file.
      * @param consumeFile if set to true, delete [[restoreFile]] after reading. */
    def restoreFromFile(consumeFile: Boolean): Option[Path] = {
      val restore = exists(restoreFile)
      if (restore) {
        val content = read(restoreFile).trim
        val p = try Path(content)
        catch {
          case _: IllegalArgumentException => pwd / RelPath(content)
        }
        if (consumeFile) {
          rm(restoreFile)
        }
        Some(p)
      } else None
    }
  }
}
