package chisel3

import chisel3.internal.firrtl._
import chisel3.internal._
import chisel3.experimental.BaseModule
import scala.reflect.ClassTag


private[chisel3] object DataMirror {
  def widthOf(target:              Data): Width = target.width
  def specifiedDirectionOf(target: Data): SpecifiedDirection = target.specifiedDirection
  def directionOf(target: Data): ActualDirection = {
    requireIsHardware(target, "node requested directionality on")
    target.direction
  }

  private def hasBinding[B <: ConstrainedBinding: ClassTag](target: Data) = {
    target.topBindingOpt match {
      case Some(b: B) => true
      case _ => false
    }
  }

  /** Check if a given `Data` is an IO port
    * @param x the `Data` to check
    * @return `true` if x is an IO port, `false` otherwise
    */
  def isIO(x: Data): Boolean = hasBinding[PortBinding](x)

  /** Check if a given `Data` is a Wire
    * @param x the `Data` to check
    * @return `true` if x is a Wire, `false` otherwise
    */
  def isWire(x: Data): Boolean = hasBinding[WireBinding](x)

  /** Check if a given `Data` is a Reg
    * @param x the `Data` to check
    * @return `true` if x is a Reg, `false` otherwise
    */
  def isReg(x: Data): Boolean = hasBinding[RegBinding](x)

  /** Check if two Chisel types are the same type.
    * Internally, this is dispatched to each Chisel type's
    * `typeEquivalent` function for each type to determine
    * if the types are intended to be equal.
    *
    * For most types, different parameters should ensure
    * that the types are different.
    * For example, `UInt(8.W)` and `UInt(16.W)` are different.
    * Likewise, Records check that both Records have the same
    * elements with the same types.
    *
    * @param x First Chisel type
    * @param y Second Chisel type
    * @return true if the two Chisel types are equal.
    */
  def checkTypeEquivalence(x: Data, y: Data): Boolean = x.typeEquivalent(y)

  /** Returns the ports of a module
    * {{{
    * class MyModule extends Module {
    *   val io = IO(new Bundle {
    *     val in = Input(UInt(8.W))
    *     val out = Output(Vec(2, UInt(8.W)))
    *   })
    *   val extra = IO(Input(UInt(8.W)))
    *   val delay = RegNext(io.in)
    *   io.out(0) := delay
    *   io.out(1) := delay + extra
    * }
    * val mod = Module(new MyModule)
    * DataMirror.modulePorts(mod)
    * // returns: Seq(
    * //   "clock" -> mod.clock,
    * //   "reset" -> mod.reset,
    * //   "io" -> mod.io,
    * //   "extra" -> mod.extra
    * // )
    * }}}
    */
  def modulePorts(target: BaseModule): Seq[(String, Data)] = target.getChiselPorts

  /** Returns a recursive representation of a module's ports with underscore-qualified names
    * {{{
    * class MyModule extends Module {
    *   val io = IO(new Bundle {
    *     val in = Input(UInt(8.W))
    *     val out = Output(Vec(2, UInt(8.W)))
    *   })
    *   val extra = IO(Input(UInt(8.W)))
    *   val delay = RegNext(io.in)
    *   io.out(0) := delay
    *   io.out(1) := delay + extra
    * }
    * val mod = Module(new MyModule)
    * DataMirror.fullModulePorts(mod)
    * // returns: Seq(
    * //   "clock" -> mod.clock,
    * //   "reset" -> mod.reset,
    * //   "io" -> mod.io,
    * //   "io_out" -> mod.io.out,
    * //   "io_out_0" -> mod.io.out(0),
    * //   "io_out_1" -> mod.io.out(1),
    * //   "io_in" -> mod.io.in,
    * //   "extra" -> mod.extra
    * // )
    * }}}
    * @note The returned ports are redundant. An [[Aggregate]] port will be present along with all
    *       of its children.
    * @see [[DataMirror.modulePorts]] for a non-recursive representation of the ports.
    */
  def fullModulePorts(target: BaseModule): Seq[(String, Data)] = {
    def getPortNames(name: String, data: Data): Seq[(String, Data)] = Seq(name -> data) ++ (data match {
      case _: Element => Seq()
      case r: Record  => r.elements.toSeq.flatMap { case (eltName, elt) => getPortNames(s"${name}_${eltName}", elt) }
      case v: Vec[_]  => v.zipWithIndex.flatMap { case (elt, index) => getPortNames(s"${name}_${index}", elt) }
    })
    modulePorts(target).flatMap {
      case (name, data) =>
        getPortNames(name, data).toList
    }
  }

  // Internal reflection-style APIs, subject to change and removal whenever.
  object internal {
    def isSynthesizable(target: Data): Boolean = target.isSynthesizable
    // For those odd cases where you need to care about object reference and uniqueness
    def chiselTypeClone[T <: Data](target: Data): T = {
      target.cloneTypeFull.asInstanceOf[T]
    }
  }


  trait HasMatchingZipOfChildren[T] {
    def matchingZipOfChildren(left: Option[T], right: Option[T]): Seq[(Option[T], Option[T])]
  }


  /** Return all expanded components, including intermediate aggregate nodes
    *
    * @param d Component to find leafs if aggregate typed. Intermediate fields/indicies ARE included
    */
  def getIntermediateAndLeafs(d: Data): Seq[Data] = d match {
    case r: Record => r +: r.getElements.flatMap(getIntermediateAndLeafs)
    case v: Vec[_] => v +: v.getElements.flatMap(getIntermediateAndLeafs)
    case other => Seq(other)
  }

  /** Collects all fields selected by collector within a data and all recursive children fields
    * Accepts a collector partial function, rather than a collector function
    *
    * @param data Data to collect fields, as well as all children datas it directly and indirectly instantiates
    * @param collector Collector partial function to pick which components to collect
    * @tparam T Type of the component that will be collected
    */
  def collectDeep[T](d: Data)(collector: PartialFunction[Data, T]): Iterable[T] = {
    val myItems = collector.lift(d)
    val deepChildrenItems = d match {
      case a: Aggregate => a.getElements.flatMap { x => collectDeep(x)(collector) }
      case other => Nil
    }
    myItems ++ deepChildrenItems
  }

  def collectAlignedDeep[T](d: Data)(pf: PartialFunction[Data, T]): Seq[T] = {
    collectDeepOverAllForAny(Some(RelativeOrientation(d, Set.empty, true)), None) {
      case (Some(x: AlignedWithRoot), _) => (pf.lift(x.data), None)
    }.map(_._1).flatten
  }

  def collectFlippedDeep[T](d: Data)(pf: PartialFunction[Data, T]): Seq[T] = {
    collectDeepOverAllForAny(Some(RelativeOrientation(d, Set.empty, true)), None) {
      case (Some(x: FlippedWithRoot), _) => (pf.lift(x.data), None)
    }.map(_._1).flatten
  }

  def collectDeepOverMatches[D : HasMatchingZipOfChildren, T](left: D, right: D)(collector: PartialFunction[(D, D), T]): Seq[T] = {
    def newCollector(lOpt: Option[D], rOpt: Option[D]): Option[(Option[T], Option[Unit])] = {
      (lOpt, rOpt) match {
        case (Some(l), Some(r)) => collector.lift((l, r)) match {
          case Some(x: T) => Some((Some(x), None))
          case None => None
        }
        case other => None
      }
    }
    collectDeepOverAllForAnyFunction(Some(left), Some(right)){
      case (Some(l), Some(r)) => collector.lift((l, r)) match {
        case Some(x: T) => Some((Some(x), None))
        case None => None
      }
      case other => None
    }.collect {
      case (Some(x: T), None) => (x)
    }
  }
  def collectDeepOverAll[D : HasMatchingZipOfChildren, T](left: D, right: D)(collector: PartialFunction[(Option[D], Option[D]), T]): Seq[T] = {
    collectDeepOverAllForAnyFunction(Some(left), Some(right)){
      case (lOpt: Option[D], rOpt: Option[D]) => collector.lift((lOpt, rOpt)) match {
        case Some(x: T) => Some((Some(x), None))
        case None => None
      }
    }.collect {
      case (Some(x: T), None) => x
    }
  }

  def collectDeepOverAllForAny[D : HasMatchingZipOfChildren, T, S](left: Option[D], right: Option[D])(pcollector: PartialFunction[(Option[D], Option[D]), (Option[T], Option[S])]): Seq[(Option[T], Option[S])] = {
    collectDeepOverAllForAnyFunction(left, right)(pcollector.lift)
  }

  def collectDeepOverAllForAnyFunction[D : HasMatchingZipOfChildren, T, S](left: Option[D], right: Option[D])(collector: ((Option[D], Option[D])) => Option[(Option[T], Option[S])]): Seq[(Option[T], Option[S])] = {
    val myItems = collector((left, right)) match {
      case None => Nil
      case Some((None, None)) => Nil
      case Some(other) => Seq(other)
    }
    val matcher = implicitly[HasMatchingZipOfChildren[D]]
    val childItems = matcher.matchingZipOfChildren(left, right).flatMap { case (l, r) => collectDeepOverAllForAnyFunction(l, r)(collector)}
    myItems ++ childItems
  }
}