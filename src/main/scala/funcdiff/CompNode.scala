package funcdiff

import botkop.numsca._
import funcdiff.DiffFunc.ConstFunc
import SimpleMath.Extensions._

import concurrent._

class CompNode(val func: DiffFunc) {
  def value: Tensor = func.value

  def shape: Array[Int] = value.shape

  def backprop: Map[CompNode, Gradient] = CompNode.backprop(this)

  def backpropForParams(
    runInParallel: Option[ExecutionContext]
  ): Map[SymbolPath, Gradient] = {
    val br = runInParallel match {
      case Some(ctx) => Await.result(this.backpropParallel(ctx), duration.Duration.Inf)
      case None      => this.backprop
    }
    br.collect { case (n: ParamNode, g) if g.nonZero => n.path -> g }
  }

  def backpropParallel(
    implicit ctx: ExecutionContext
  ): Future[Map[CompNode, Gradient]] = CompNode.backpropParallel(this)

  override def toString: String = {
    s"Node[name=${func.name}, shape=${TensorExtension.showShape(shape)}, value=$value]"
  }

  def withShape(shape: Int*): CompNode = {
    assert(
      this.shape sameElements shape,
      s"expected shape: ${TensorExtension.showShape(shape.toArray)}, " +
        s"actual: ${TensorExtension.showShape(this.shape)}, node: $this"
    )
    this
  }
}

class ParamNode(v: Tensor, val path: SymbolPath) extends CompNode(ConstFunc(v)) {

  override def toString: String = {
    s"param{$path, shape=${TensorExtension.showShape(value.shape)}"
  }
}

object CompNode {

  /**
    * Back-propagate gradients through the computation graph
    */
  def backprop(nodes: List[CompNode], grads: List[Gradient]): Map[CompNode, Gradient] = {
    import collection.mutable
    assert(nodes.length == grads.length)

    val sorted = topologicalSort(nodes)
    val gBuilders = grads.map { g =>
      new GradientBuilder(g, needCopy = true)
    }
    val gradients = mutable.HashMap[CompNode, GradientBuilder](nodes.zip(gBuilders): _*)

    for {
      n <- sorted
      gBuilder <- gradients.get(n)
      argGradients = n.func.backProp(gBuilder.retrieve)
      (arg, grad) <- n.func.args.zip(argGradients) if !grad.isZero
    } {

      val aG = gradients.getOrElseUpdate(
        arg,
        new GradientBuilder(ZeroGradient(arg.shape), needCopy = false)
      )
//      val aG = gradients.getOrElse(arg, DenseGradient(numsca.zeros(arg.shape)))
      aG.add(grad)
    }

    assert(
      grads.any(_.isZero) || !gradients.values.exists(_.retrieve.isZero),
      "nonempty gradients after backprop created empty gradients!"
    )
    gradients.toMap.mapValuesNow(_.retrieve)
  }

  def backprop(node: CompNode): Map[CompNode, Gradient] = {
    backprop(List(node), List(DenseGradient(ones(node.shape))))
  }

  import collection.mutable

  /** Count the number of child nodes for each CompNode */
  def countChildren(nodes: List[CompNode]): mutable.HashMap[CompNode, Int] = {
    import collection.mutable
    val nodeCounters = mutable.HashMap[CompNode, Int]()

    def setupCounter(n: CompNode): Unit = {
      nodeCounters.get(n) match {
        case None =>
          nodeCounters(n) = 1
          n.func.args.foreach(setupCounter)
        case Some(c) =>
          nodeCounters(n) = c + 1
      }
    }
    nodes.foreach(setupCounter)
    nodeCounters
  }

  /**
    * Returns a reverse-topologically sorted list of the nodes. The first element corresponds
    * to the first node we should perform backprop on.
    */
  def topologicalSort(
    nodes: List[CompNode],
    childNumbers: Option[mutable.Map[CompNode, Int]] = None
  ): List[CompNode] = {
    val nodeCounters = childNumbers.getOrElse(countChildren(nodes))

    def rec(sorted: List[CompNode], left: List[CompNode]): List[CompNode] = {
      left match {
        case h :: tl =>
          var newLeft = tl
          h.func.args.foreach { n =>
            nodeCounters(n) -= 1
            if (nodeCounters(n) == 0) {
              newLeft = n :: newLeft
            }
          }
          rec(h :: sorted, newLeft)
        case List() => sorted
      }
    }

    rec(List(), nodes).reverse
  }

  import concurrent._

  /**
    * Back-propagate gradients through the computation graph in parallel
    */
  def backpropParallel(nodes: List[CompNode], grads: List[Gradient])(
    implicit ctx: ExecutionContext
  ): Future[Map[CompNode, Gradient]] = {
    import collection.mutable
    assert(nodes.length == grads.length)

    val childrenNumber = countChildren(nodes)

    class Counter(var i: Int)
    val childrenCounter = childrenNumber.map{ case (n, i) => n -> new Counter(i)}

    class GradientHolder(val builder: GradientBuilder) {
      var currentTask: Future[Unit] = Future.successful()
      def add(g: Gradient): Future[Unit] = synchronized {
        val fut = currentTask.map { _ =>
          builder.add(g)
        }
        currentTask = fut
        fut
      }

      def get: Future[Gradient] = {
        currentTask.map { _ =>
          builder.retrieve
        }
      }
    }

    val gradients = DebugTime.logTime('gradsInit) {
      mutable.HashMap(topologicalSort(nodes, Some(childrenNumber)).map { n =>
        n -> new GradientHolder(
          new GradientBuilder(ZeroGradient(n.shape), needCopy = false)
        )
      }: _*)
    }
    nodes.zip(grads).foreach { case (n, g) => gradients(n).builder.add(g) }

    def rec(node: CompNode): Future[Unit] = {
      val cn = {
        val c = childrenCounter(node)
        c.synchronized {
          c.i -= 1
          c.i
        }
      }

      if (cn > 0) return Future.successful(())
      gradients.get(node) match {
        case None => Future.successful(Unit)
        case Some(holder) =>
          holder.get.map(node.func.backProp).flatMap { argGradients =>
            val futs = for {
              (arg, grad) <- node.func.args.zip(argGradients) if !grad.isZero
            } yield {
              gradients(arg).add(grad).map(_ => rec(arg))
            }
            Future.sequence(futs).map(_ => ())
          }
      }
    }

    Future.traverse(nodes)(rec).flatMap { _ =>
      Future
        .sequence(gradients.toSeq.map {
          case (k, gB) =>
            gB.get.map { g =>
              k -> g
            }
        })
        .map { pairs =>
          pairs.filter(_._2.nonZero).toMap
        }
    }
  }

  def backpropParallel(node: CompNode)(
    implicit ctx: ExecutionContext
  ): Future[Map[CompNode, Gradient]] = {
    backpropParallel(List(node), List(DenseGradient(ones(node.shape))))
  }

  def defaultNodeName(n: CompNode): String = n match {
    case p: ParamNode => SimpleMath.wrapInQuotes(p.path.toString)
    case _            => SimpleMath.wrapInQuotes(n.func.name)
  }

  def visualize(
    node: CompNode,
    nodeInfo: CompNode => String = n => SimpleMath.wrapInQuotes(n.func.name),
    nodeName: CompNode => String = defaultNodeName
  ): String = {
    var id = 0
    val idMap = mutable.HashMap[CompNode, Int]()
    val connectivity = mutable.HashMap[Int, Int]()
    val nodeMessages = mutable.ListBuffer[String]()

    def rec(n: CompNode): Unit = {
      assert(!idMap.contains(n))
      val thisId = id
      id += 1
      idMap(n) = thisId
      nodeMessages += s"Labeled[$id, Tooltip[${nodeName(n)}, ${nodeInfo(n)}]]"
      for (a <- n.func.args) {
        if (!idMap.contains(a))
          rec(a)
        connectivity(idMap(a)) = thisId
      }
    }
    rec(node)

    val edgePart = connectivity.map { case (a, b) => s"$a->$b" }.mkString("{", ",", "}")
    s"""Graph[${nodeMessages.mkString("{", ",", "}")},$edgePart,GraphLayout -> "LayeredDigraphEmbedding"]"""
  }

}
