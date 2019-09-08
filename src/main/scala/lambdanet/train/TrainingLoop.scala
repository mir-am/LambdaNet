package lambdanet.train

import java.util.Calendar

import lambdanet._
import java.util.concurrent.ForkJoinPool

import botkop.numsca
import cats.Monoid
import funcdiff.{SimpleMath => SM}
import funcdiff._
import lambdanet.architecture._
import lambdanet.utils.{EventLogger, QLangAccuracy, QLangDisplay, ReportFinish}
import TrainingState._
import botkop.numsca.Tensor
import lambdanet.architecture.LabelEncoder.{
  SegmentedLabelEncoder,
  TrainableLabelEncoder
}
import lambdanet.translation.PredicateGraph.{PNode, PType, ProjNode}
import org.nd4j.linalg.api.buffer.DataType

import scala.collection.parallel.ForkJoinTaskSupport
import scala.concurrent.{
  Await,
  ExecutionContext,
  ExecutionContextExecutorService,
  Future,
  TimeoutException
}
import scala.language.reflectiveCalls
import scala.util.Random

object TrainingLoop extends TrainingLoopTrait {
  val toyMod: Boolean = false
  val onlySeqModel = false
  val useDropout: Boolean = true
  val useOracleForIsLib: Boolean = true
  /* Assign more weights to project type to battle label imbalance */
  val maxLibRatio: Real = 3.0
  val projWeight: Real = maxLibRatio

  val taskName: String = {
    val flags = Seq(
      "oracle" -> useOracleForIsLib,
      "fix" -> NeuralInference.fixBetweenIteration,
      "toy" -> toyMod,
      "weighted" -> (projWeight != 1.0)
    ).map(flag).mkString

    if (onlySeqModel) "large-seqModel"
    else s"attendPS-fc1-newSim" + s"$flags-${TrainingState.iterationNum}"
  }

  def flag(nameValue: (String, Boolean)): String = {
    val (name, value) = nameValue
    if (value) s"-$name" else ""
  }

  import fileLogger.{println, printInfo, printWarning, printResult, announced}

  def scaleLearningRate(epoch: Int): Double = {
    val min = 0.3
    val epochToSlowDown = if (toyMod) 100 else 30
    SimpleMath
      .linearInterpolate(1.0, min)(epoch.toDouble / epochToSlowDown)
      .max(min)
  }

  def main(args: Array[String]): Unit = {
    Tensor.floatingDataType = DataType.DOUBLE

    run(
      maxTrainingEpochs = if (toyMod) 1000 else 200,
      numOfThreads = readThreadNumber()
    ).result()
  }

  case class run(
      maxTrainingEpochs: Int,
      numOfThreads: Int
  ) {

    printInfo(s"Task: $taskName")
    printInfo(s"maxEpochs = $maxTrainingEpochs, threads = $numOfThreads")
    Timeouts.readFromFile()

    def result(): Unit = {
      val (state, pc, logger) = loadTrainingState(resultsDir, fileLogger)
      val architecture = GruArchitecture(state.dimMessage, pc)
      printResult(s"NN Architecture: ${architecture.arcName}")
      printResult(s"Single layer consists of: ${architecture.messageLayerModel}")
      val seqArchitecture =
        SequenceModel.SeqArchitecture(state.dimMessage, pc)
      val dataSet =
        DataSet.loadDataSet(taskSupport, architecture, toyMod, maxLibRatio)

//      NamingBaseline.test(dataSet)

      val run = runOnProjects(
        dataSet,
        state,
        pc,
        logger,
        architecture,
        seqArchitecture
      ).train()

//      namingHelpfulness(dataSet, run)
    }

    def namingHelpfulness(dataSet: DataSet, run: runOnProjects) = {
      import cats.implicits._

      def showCount(c: Counted[Int]): String = {
        s"$c, percentage: ${toAccuracy(c)}"
      }

      val (helpSet, notHelpSet) =
        (for {
          datum <- dataSet.trainSet
          predictions = NamingBaseline
            .testOnDatum(datum, useOracle = true, identity)
            .predict
          fwd <- run
            .forward(
              datum,
              shouldDownsample = false,
              shouldDropout = false,
              maxBatchSize = Some(600)
            )
            .map(_._2)
          Seq(s1, s2) = Seq(fwd.incorrectSet -> false, fwd.correctSet -> true)
            .map {
              case (set, correctness) =>
                set
                  .filterNot(_._2.madeFromLibTypes)
                  .toVector
                  .foldMap {
                    case (n, label, _) =>
                      val (truthPosition, label1) = predictions(ProjNode(n))
                      assert(label1 == label)
                      Counted.fromBool((truthPosition == 0) != correctness)
                  }
                  .tap { stat =>
                    val name = if (correctness) "unhelpful" else "help"
                    printResult(s"$name set: ${showCount(stat)}")
                  }
            }
        } yield (s1, s2)).combineAll

      printResult(s"Overall helpful set: ${showCount(helpSet)}")
      printResult(s"Overall unhelpful set: ${showCount(notHelpSet)}")
    }

    //noinspection TypeAnnotation
    case class runOnProjects(
        dataSet: DataSet,
        trainingState: TrainingState,
        pc: ParamCollection,
        logger: EventLogger,
        architecture: NNArchitecture,
        seqArchitecture: SequenceModel.SeqArchitecture
    ) {
      import dataSet._
      import trainingState._

      val maxBatchSize = dataSet
        .signalSizeMedian(maxLibRatio)
        .tap(s => printResult(s"maxBatchSize: $s"))

      val rand = new Random(1)
      def randomLabelId(): Int = rand.synchronized {
        rand.nextInt(50)
      }
      val Seq(labelEncoder, nameEncoder) =
        Seq("labelEncoder", "nameEncoder").map { name =>
          SegmentedLabelEncoder(
            name,
            trainSet,
            coverageGoal = 0.98,
            architecture,
            dropoutProb = 0.1,
            dropoutThreshold = 1000,
            randomLabelId
          )
        }
      printResult(s"Label encoder: ${labelEncoder.name}")
      printResult(s"Name encoder: ${nameEncoder.name}")

      val labelCoverage =
        TrainableLabelEncoder(
          trainSet,
          coverageGoal = 0.90,
          architecture,
          dropoutProb = 0.1,
          dropoutThreshold = 500,
          randomLabelId
        )

      var shouldAnnounce: Boolean = true

      def train(): Unit = {
        val saveInterval = if (toyMod) 40 else 6

        (trainingState.epoch0 + 1 to maxTrainingEpochs).foreach { epoch =>
          shouldAnnounce = epoch == 1 // only announce in the first epoch for debugging purpose
          announced(s"epoch $epoch") {
            TensorExtension.checkNaN = false // (epoch - 1) % 10 == 0
            handleExceptions(epoch) {
              trainStep(epoch)
              if ((epoch - 1) % 3 == 0)
                DebugTime.logTime("testSteps") {
                  testStep(epoch, isTestSet = false)
                  testStep(epoch, isTestSet = true)
                }
              if (epoch == 1 || epoch % saveInterval == 0)
                DebugTime.logTime("saveTraining") {
                  saveTraining(epoch, s"epoch$epoch")
                }
            }
          }
        }

        saveTraining(maxTrainingEpochs, "finished")
        emailService.sendMail(emailService.userEmail)(
          s"TypingNet: Training finished on $machineName!",
          "Training finished!"
        )
      }

      val (machineName, emailService) = ReportFinish.readEmailInfo(taskName)
      private def handleExceptions(epoch: Int)(f: => Unit): Unit = {
        try f
        catch {
          case ex: Throwable =>
            val isTimeout = ex.isInstanceOf[TimeoutException]
            val errorName = if (isTimeout) "timeout" else "stopped"
            emailService.sendMail(emailService.userEmail)(
              s"TypingNet: $errorName on $machineName at epoch $epoch",
              s"Details:\n" + ex.getMessage
            )
            if (isTimeout && Timeouts.restartOnTimeout) {
              printWarning(
                "Timeout... training restarted (skip one training epoch)..."
              )
            } else {
              if (!ex.isInstanceOf[StopException]) {
                ex.printStackTrace()
                saveTraining(epoch, "error-save", skipTest = true)
              }
              throw ex
            }
        }
      }

      val random = new util.Random(2)

      def trainStep(epoch: Int): Unit = {
        DebugTime.logTime("GC") {
          System.gc()
        }

        val startTime = System.nanoTime()
        val oldOrder = random.shuffle(trainSet)
        val (h, t) = oldOrder.splitAt(119)
        val stats = (t ++ h).zipWithIndex.map {
          case (datum, i) =>
            import Console.{GREEN, BLUE}
            announced(
              s"$GREEN[epoch $epoch](progress: ${i + 1}/${trainSet.size})$BLUE train on $datum",
              shouldAnnounce
            ) {
//              println(DebugTime.show)
              checkShouldStop(epoch)
              for {
                (loss, fwd, _) <- forward(
                  datum,
                  shouldDownsample = true,
                  shouldDropout = useDropout,
                  maxBatchSize = Some(maxBatchSize)
                ).tap(
                  _.foreach(r => printResult(r._2))
                )
                _ = checkShouldStop(epoch)
              } yield {
                checkShouldStop(epoch)
                def optimize(loss: CompNode) = {
                  val factor = fwd.loss.count.toDouble / avgAnnotations
                  optimizer.minimize(
                    loss * factor,
                    pc.allParams,
                    backPropInParallel =
                      Some(parallelCtx -> Timeouts.optimizationTimeout),
                    gradientTransform = _.clipNorm(2 * factor),
                    scaleLearningRate = scaleLearningRate(epoch)
                  )
                }

                val gradInfo = limitTimeOpt(
                  s"optimization: $datum",
                  Timeouts.optimizationTimeout
                ) {
                  announced("optimization", shouldAnnounce) {
                    val stats = DebugTime.logTime("optimization") {
                      optimize(loss)
                    }
                    calcGradInfo(stats)
                  }
                }.toVector

                println(DebugTime.show)
                (fwd, gradInfo, datum)
              }
            }
        }

        import cats.implicits._
        val (fws, gs, data) = stats.flatMap(_.toVector).unzip3

        import logger._

        fws.combineAll.tap { fwd =>
          import fwd._
          logScalar("loss", epoch, toAccuracyD(loss))
          logScalar("libAcc", epoch, toAccuracy(libCorrect))
          logScalar("projAcc", epoch, toAccuracy(projCorrect))
          logConfusionMatrix("confusionMat", epoch, confusionMatrix.value, 2)
          logAccuracyDetails(data zip fws, epoch)
        }

        val gradInfo = gs.combineAll
        gradInfo.unzip3.tap {
          case (grads, transformed, deltas) =>
            logScalar("gradient", epoch, grads.sum)
            logScalar("clippedGradient", epoch, transformed.sum)
            logScalar("paramDelta", epoch, deltas.sum)
        }

        val timeInSec = (System.nanoTime() - startTime).toDouble / 1e9
        logScalar("iter-time", epoch, timeInSec)

        println(DebugTime.show)
      }

      private def typeAccString(accs: Map[PType, Counted[Correct]]): String = {
        val (tys, counts) = accs.toVector.sortBy { c =>
          -c._2.count
        }.unzip
        val typeStr = tys
          .map(t => SM.wrapInQuotes(t.showSimple))
          .mkString("{", ",", "}")
        val countStr = counts
          .map(c => s"{${c.count}, ${c.value}}")
          .mkString("{", ",", "}")
        s"{$typeStr,$countStr}"
      }

      private def logAccuracyDetails(
          stats: Vector[(Datum, ForwardResult)],
          epoch: Int
      ) = {
        import cats.implicits._
        val str = stats
          .map {
            case (d, f) =>
              val size = d.predictor.graph.predicates.size
              val acc = toAccuracy(
                f.libCorrect.combine(f.projCorrect)
              )
              val name = d.projectName
              s"""{$size, $acc, "$name"}"""
          }
          .mkString("{", ",", "}")
        logger.logString("accuracy-distr", epoch, str)
      }

      def testStep(epoch: Int, isTestSet: Boolean): Unit = {
        val dataSetName = if (isTestSet) "test" else "dev"
        val dataSet = if (isTestSet) testSet else devSet
        announced(s"test on $dataSetName set") {
          import cats.implicits._

          val (stat, fse1Acc, projTop5Acc) = dataSet.flatMap { datum =>
            checkShouldStop(epoch)
            announced(s"test on $datum", shouldAnnounce) {
              forward(
                datum,
                shouldDownsample = !isTestSet,
                shouldDropout = false,
                maxBatchSize = Some(maxBatchSize)
              ).map {
                case (_, fwd, pred) =>
                  val (fse1, _, _) = datum.fseAcc
                    .countTopNCorrect(
                      1,
                      pred.mapValuesNow(_.distr.map(_._2)),
                      onlyCountInSpaceTypes = true
                    )
                  val projTop5 = {
                    val predictions = pred.map {
                      case (n, distr) => n -> distr.distr.take(5).map(_._2)
                    }
                    val nodesMap = datum.nodesToPredict.collect {
                      case (n, ty)
                          if predictions.contains(n.n) && !ty.madeFromLibTypes =>
                        n.n -> ty
                    }
                    QLangAccuracy
                      .countTopNCorrect(
                        5,
                        nodesMap,
                        predictions,
                        _ => 1
                      )
                      ._1
                  }
                  (fwd, fse1, projTop5)

              }.toVector
            }
          }.combineAll

          import stat.{libCorrect, projCorrect, confusionMatrix, categoricalAcc}
          import logger._
          logScalar(s"$dataSetName-loss", epoch, toAccuracyD(stat.loss))
          logScalar(s"$dataSetName-libAcc", epoch, toAccuracy(libCorrect))
          logScalar(s"$dataSetName-projAcc", epoch, toAccuracy(projCorrect))
          logScalar(s"$dataSetName-projTop5Acc", epoch, toAccuracy(projTop5Acc))
          logConfusionMatrix(
            s"$dataSetName-confusionMat",
            epoch,
            confusionMatrix.value,
            2
          )
          logScalar(s"$dataSetName-fse-top1", epoch, toAccuracy(fse1Acc))
          val libTypeAcc = categoricalAcc.filter(_._1.madeFromLibTypes)
          logString(
            s"$dataSetName-typeAcc",
            epoch,
            typeAccString(libTypeAcc)
          )
          val projTypeAcc = categoricalAcc.filterNot(_._1.madeFromLibTypes)
          logString(
            s"$dataSetName-proj-typeAcc",
            epoch,
            typeAccString(projTypeAcc)
          )
        }
      }

      def calcGradInfo(stats: Optimizer.OptimizeStats) = {
        def meanSquaredNorm(gs: Iterable[Gradient]) = {
          import numsca._
          import cats.implicits._
          val combined = gs.toVector.map { g =>
            val t = g.toTensor()
            Counted(t.elements.toInt, sum(square(t)))
          }.combineAll
          math.sqrt(combined.value / nonZero(combined.count))
        }

        val grads = meanSquaredNorm(stats.gradients.values)
        val transformed = meanSquaredNorm(stats.transformedGrads.values)
        val deltas = meanSquaredNorm(stats.deltas.values)
        (grads, transformed, deltas)
      }

      val lossModel: LossModel = LossModel.NormalLoss
        .tap(m => printResult(s"loss model: ${m.name}"))

//      /** Forward propagation for the sequential model */
//      private def seqForward(
//          datum: Datum
//      ): Option[(Loss, ForwardResult, Map[PNode, TopNDistribution[PType]])] = {
//        def result = {
//          val predictor = datum.seqPredictor
//          val predSpace = predictor.predSpace
//          // the logits for very iterations
//          val nodes = datum.nodesToPredict.map { _.n }
//          val logits = announced("run seq predictor") {
//            predictor.run(
//              seqArchitecture,
//              nameEncoder,
//              nodes,
//              nameDropout = useDropout && isTraining
//            )
//          }
//
//          val nonGenerifyIt = DataSet.nonGenerify(predictor.libDefs)
//
//          val groundTruths = nodes.map {
//            case n if n.fromLib =>
//              nonGenerifyIt(predictor.libDefs.nodeMapping(n).get)
//            case n if n.fromProject =>
//              datum.nodesToPredict(ProjNode(n))
//          }
//
//          val targets = groundTruths.map(predSpace.indexOfType)
//          val nodeDistances = nodes.map(_.pipe(datum.distanceToConsts))
//
//          val (correctness, confMat, typeAccs) =
//            announced("compute training accuracy") {
//              analyzeDecoding(
//                logits,
//                groundTruths,
//                predSpace,
//                nodeDistances
//              )
//            }
//
//          val loss =
//            logits.toLoss(targets, projWeight, predSpace.libTypeVec.length)
//
//          val totalCount = groundTruths.length
//          val mapped = nodes
//            .zip(groundTruths)
//            .zip(correctness)
//            .groupBy(_._2)
//            .mapValuesNow { pairs =>
//              pairs.map { case ((n, ty), _) => (n, ty, datum.projectName) }.toSet
//            }
//          val fwd = ForwardResult(
//            Counted(totalCount, loss.value.squeeze() * totalCount),
//            mapped.getOrElse(true, Set()),
//            mapped.getOrElse(false, Set()),
//            confMat,
//            typeAccs
//          )
//
//          val predictions = {
//            val predVec = logits
//              .topNPredictionsWithCertainty(6)
//              .map { _.map(predSpace.typeVector) }
//            nodes.zip(predVec).toMap
//          }
//
//          (loss, fwd, predictions)
//        }
//
//        limitTimeOpt(s"forward: $datum", Timeouts.forwardTimeout) {
//          DebugTime.logTime("seqForward") { result }
//        }
//      }
      def forward(
          datum: Datum,
          shouldDownsample: Boolean,
          shouldDropout: Boolean,
          maxBatchSize: Option[Int]
      ): Option[(Loss, ForwardResult, Map[PNode, TopNDistribution[PType]])] =
        limitTimeOpt(s"forward: $datum", Timeouts.forwardTimeout) {
          import datum._

          val predSpace = predictor.predictionSpace

          val annotsToUse =
            if (shouldDownsample) datum.downsampleLibAnnots(maxLibRatio, random)
            else nodesToPredict

          val (nodes, groundTruths) = annotsToUse.toVector
            .pipe(random.shuffle(_))
            .take(maxBatchSize.getOrElse(Int.MaxValue))
            .unzip
          val targets = groundTruths.map(predSpace.indexOfType)
          val isLibOracle =
            if (useOracleForIsLib) Some(targets.map(predSpace.isLibType))
            else None

          val decodingVec = announced("run predictor", shouldAnnounce) {
            predictor
              .run(
                architecture,
                nodes,
                iterationNum,
                labelEncoder,
                labelCoverage.isLibLabel,
                nameEncoder,
                shouldDropout,
                shouldDropout,
                isLibOracle
              )
              .result
          }
          val decoding = decodingVec.last

          val (correctness, confMat, typeAccs) =
            announced("compute training accuracy", shouldAnnounce) {
              analyzeDecoding(
                decoding,
                groundTruths,
                predSpace
              )
            }

          val loss = lossModel.predictionLoss(
            predictor
              .parallelize(decodingVec)
              .map(_.toLoss(targets, projWeight, predSpace.libTypeVec.length))
          )

          val totalCount = groundTruths.length
          val mapped = nodes
            .map(_.n)
            .zip(groundTruths)
            .zip(correctness)
            .groupBy(_._2)
            .mapValuesNow { pairs =>
              pairs.map { case ((n, ty), _) => (n, ty, datum.projectName) }.toSet
            }

          val fwd = ForwardResult(
            Counted(totalCount, loss.value.squeeze() * totalCount),
            mapped.getOrElse(true, Set()),
            mapped.getOrElse(false, Set()),
            confMat,
            typeAccs
          ).tap(r => assert(r.isConsistent))

          val predictions = {
            val predVec = decoding
              .topNPredictionsWithCertainty(6)
              .map { _.map(predSpace.typeOfIndex) }
            nodes.map(_.n).zip(predVec).toMap
          }

          (loss, fwd, predictions)
        }

      @throws[TimeoutException]
      private def limitTime[A](timeLimit: Timeouts.Duration)(f: => A): A = {
        val exec = scala.concurrent.ExecutionContext.global
        Await.result(Future(f)(exec), timeLimit)
      }

      private def limitTimeOpt[A](
          name: String,
          timeLimit: Timeouts.Duration
      )(f: => A): Option[A] = {
        try {
          Some(limitTime(timeLimit)(f))
        } catch {
          case _: TimeoutException =>
            val msg = s"$name exceeded time limit $timeLimit."
            printWarning(msg)
            emailService.atFirstTime {
              emailService.sendMail(emailService.userEmail)(
                s"TypingNet: timeout on $machineName during $name",
                s"Details:\n" + msg
              )
            }
            None
        }
      }

      import ammonite.ops._

      private def saveTraining(
          epoch: Int,
          dirName: String,
          skipTest: Boolean = false
      ): Unit = {
        announced(s"save training to $dirName") {
          val saveDir = resultsDir / "saved" / dirName
          if (!exists(saveDir)) {
            mkdir(saveDir)
          }
          // do the following tasks in parallel
          val tasks = Vector(
            () => {
              val savePath = saveDir / "state.serialized"
              TrainingState(epoch, dimMessage, iterationNum, optimizer)
                .saveToFile(savePath)
            },
            () => {
              pc.saveToFile(saveDir / "params.serialized")
            },
            () => {
              val currentLogFile = resultsDir / "log.txt"
              if (exists(currentLogFile)) {
                cp.over(currentLogFile, saveDir / "log.txt")
              }
            },
            () =>
              if (testSet.nonEmpty && !skipTest) {
                import cats.implicits._

                var progress = 0
                val (right, wrong) = testSet.flatMap {
                  datum =>
                    checkShouldStop(epoch)
                    announced(
                      s"(progress: ${progress.tap(_ => progress += 1)}) test on $datum",
                      shouldAnnounce = true
                    ) {
                      forward(
                        datum,
                        shouldDownsample = true, // fixme: this is for saving training time
                        shouldDropout = false,
                        maxBatchSize = None,
                      ).map {
                        case (_, fwd, pred) =>
                          DebugTime.logTime("printQSource") {
                            QLangDisplay.renderProjectToDirectory(
                              datum.projectName.toString,
                              datum.qModules,
                              pred,
                              datum.predictor.predictionSpace.allTypes
                            )(saveDir / "predictions")
                          }
                          (fwd.correctSet, fwd.incorrectSet)
                      }.toVector
                    }
                }.combineAll

                QLangDisplay.renderPredictionIndexToDir(
                  right,
                  wrong,
                  saveDir,
                  sourcePath = "predictions"
                )
              }
          )

          tasks.par.foreach(_.apply())

          val dateTime = Calendar.getInstance().getTime
          write.over(saveDir / "time.txt", dateTime.toString)
        }
      }

      @throws[StopException]
      private def checkShouldStop(epoch: Int): Unit = {
        if (TrainingControl(resultsDir).shouldStop(consumeFile = true)) {
          saveTraining(epoch, s"stopped-epoch$epoch")
          throw StopException("Stopped by 'stop.txt'.")
        }
      }

      private def analyzeDecoding(
          results: DecodingResult,
          groundTruths: Vector[PType],
          predictionSpace: PredictionSpace
      ): (
          Vector[Boolean],
          Counted[ConfusionMatrix],
          Map[PType, Counted[Correct]]
      ) = {
        val predictions = results.topPredictions
        val targets = groundTruths.map(predictionSpace.indexOfType)
        val correctness = predictions.zip(targets).map { case (x, y) => x == y }
        val targetFromLibrary = groundTruths.map { _.madeFromLibTypes }

        val confMat = {
          def toCat(isLibType: Boolean): Int = if (isLibType) 0 else 1
          val predictionCats = predictions.map { i =>
            toCat(predictionSpace.isLibType(i))
          }
          val truthCats = targetFromLibrary.map(toCat)
          val mat = confusionMatrix(predictionCats, truthCats, categories = 2)
          Counted(predictionCats.length, mat)
        }

        val typeAccs =
          groundTruths.zip(correctness).groupBy(_._1).mapValuesNow { bools =>
            Counted(bools.length, bools.count(_._2))
          }

        (correctness, confMat, typeAccs)
      }

      private val avgAnnotations =
        SM.mean(
          trainSet.map(_.downsampleLibAnnots(maxLibRatio, random).size.toDouble)
        )
    }

    val taskSupport: Option[ForkJoinTaskSupport] =
      if (numOfThreads == 1) None
      else Some(new ForkJoinTaskSupport(new ForkJoinPool(numOfThreads)))
    val parallelCtx: ExecutionContextExecutorService = {
      import ExecutionContext.fromExecutorService
      fromExecutorService(new ForkJoinPool(numOfThreads))
    }
  }

  case class ForwardResult(
      loss: Counted[Double],
      correctSet: Set[(PNode, PType, ProjectPath)],
      incorrectSet: Set[(PNode, PType, ProjectPath)],
      confusionMatrix: Counted[ConfusionMatrix],
      categoricalAcc: Map[PType, Counted[Correct]]
  ) {
    override def toString: String = {
      s"forward result: {loss: ${toAccuracyD(loss)}, " +
        s"lib acc: ${toAccuracy(libCorrect)} (${libCorrect.count} nodes), " +
        s"proj acc: ${toAccuracy(projCorrect)} (${projCorrect.count} nodes)}"
    }

    private def countCorrect(isLibType: Boolean) = {
      val correct = correctSet.count(_._2.madeFromLibTypes == isLibType)
      val incorrect = incorrectSet.count(_._2.madeFromLibTypes == isLibType)
      Counted(correct + incorrect, correct)
    }

    def libCorrect: Counted[LibCorrect] = countCorrect(true)
    def projCorrect: Counted[ProjCorrect] = countCorrect(false)

    def isConsistent: Boolean = {
      categoricalAcc.keySet == (correctSet ++ incorrectSet).map(_._2)
    }
  }

  private implicit val forwardResultMonoid: Monoid[ForwardResult] =
    new Monoid[ForwardResult] {
      import Counted.zero
      import cats.implicits._

      def empty: ForwardResult =
        ForwardResult(zero(0), Set(), Set(), zero(Map()), Map())

      def combine(x: ForwardResult, y: ForwardResult): ForwardResult = {
        val z = ForwardResult.unapply(x).get |+| ForwardResult
          .unapply(y)
          .get
        (ForwardResult.apply _).tupled(z)
      }
    }

}
