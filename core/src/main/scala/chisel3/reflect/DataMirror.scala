// SPDX-License-Identifier: Apache-2.0

package chisel3.reflect

import chisel3._
import chisel3.internal._
import chisel3.internal.firrtl._
import chisel3.experimental.{BaseModule, SourceInfo}
import scala.reflect.ClassTag

object DataMirror {
  def widthOf(target:              Data): Width = target.width
  def specifiedDirectionOf(target: Data): SpecifiedDirection = target.specifiedDirection
  def directionOf(target: Data): ActualDirection = {
    requireIsHardware(target, "node requested directionality on")
    target.direction
  }

  /** Returns true if target has been `Flipped` or `Input` directly */
  def hasOuterFlip(target: Data): Boolean = {
    import chisel3.SpecifiedDirection.{Flip, Input}
    target.specifiedDirection match {
      case Flip | Input => true
      case _            => false
    }
  }

  private def hasBinding[B <: ConstrainedBinding: ClassTag](target: Data) = {
    // Cannot use isDefined because of the ClassTag
    target.topBindingOpt match {
      case Some(b: B) => true
      case _ => false
    }
  }

  /** Check if a given `Data` is an IO port
    * @param x the `Data` to check
    * @return `true` if x is an IO port, `false` otherwise
    */
  def isIO(x: Data): Boolean = hasBinding[PortBinding](x) || hasBinding[SecretPortBinding](x)

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

  /** Check if a given `Data` is a Probe
    * @param x the `Data` to check
    * @return `true` if x is a Probe, `false` otherwise
    */
  def hasProbeTypeModifier(x: Data): Boolean = x.probeInfo.nonEmpty

  /** Get an early guess for the name of this [[Data]]
    *
    * '''Warning: it is not guaranteed that this name will end up in the output FIRRTL or Verilog.'''
    *
    * Name guesses are not stable and may change due to a subsequent [[Data.suggestName]] or
    * plugin-related naming.
    * Name guesses are not necessarily legal Verilog identifiers.
    * Name guesses for elements of Bundles or Records will include periods, and guesses for elements
    * of Vecs will include square brackets.
    */
  def queryNameGuess(x: Data): String = {
    requireIsHardware(x, "To queryNameGuess,")
    x.earlyName
  }

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
    * Equivalent to being structural, alignment, and width type equivalent
    *
    * @param x First Chisel type
    * @param y Second Chisel type
    * @return true if the two Chisel types are equal.
    */
  def checkTypeEquivalence(x: Data, y: Data): Boolean = x.typeEquivalent(y)

  /** Check if two Chisel types have the same alignments for all matching members
    *
    * This means that for matching members in Aggregates, they must have matching member alignments relative to the parent type
    * For matching non-aggregates, they must be the same alignment to their parent type.
    *
    * @param x First Chisel type
    * @param y Second Chisel type
    * @return true if the two Chisel types have alignment type equivalence.
    */
  def checkAlignmentTypeEquivalence(x: Data, y: Data): Boolean = {
    //TODO(azidar): Perhaps there is a better pattern of `iterateOverMatches` that we can support
    collectMembersOverMatches(connectable.Alignment(x, true), connectable.Alignment(y, true)) {
      case (a, b) => a.alignment == b.alignment
    }(AlignmentMatchingZipOfChildren).forall(r => r)
  }

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
  def modulePorts(target: BaseModule)(implicit si: SourceInfo): Seq[(String, Data)] = target.getChiselPorts.collect {
    case (name, port: Data) => (name, port)
  }

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
  def fullModulePorts(target: BaseModule)(implicit si: SourceInfo): Seq[(String, Data)] = {
    def getPortNames(name: String, data: Data): Seq[(String, Data)] = Seq(name -> data) ++ (data match {
      case _: Element => Seq()
      case r: Record =>
        r._elements.toSeq.flatMap {
          case (eltName, elt) =>
            if (r._isOpaqueType) { getPortNames(s"${name}", elt) }
            else { getPortNames(s"${name}_${eltName}", elt) }
        }
      case v: Vec[_] => v.zipWithIndex.flatMap { case (elt, index) => getPortNames(s"${name}_${index}", elt) }
    })
    modulePorts(target).flatMap {
      case (name, data) =>
        getPortNames(name, data).toList
    }
  }

  /** Returns the parent module within which a module instance is instantiated
    *
    * @note Top-level modules in any given elaboration do not have a parent
    * @param target a module instance
    * @return the parent of the `target`, if one exists
    */
  def getParent(target: BaseModule): Option[BaseModule] = target._parent

  // Internal reflection-style APIs, subject to change and removal whenever.
  object internal {
    def isSynthesizable(target: Data): Boolean = target.isSynthesizable
    // For those odd cases where you need to care about object reference and uniqueness
    def chiselTypeClone[T <: Data](target: T): T = {
      target.cloneTypeFull
    }
  }

  // Old definition of collectLeafMembers
  @deprecated("Use DataMirror.collectLeafMembers instead")
  def getLeafs(d: Data): Seq[Data] = collectLeafMembers(d)

  // Old definition of collectAllChildren
  @deprecated("Use DataMirror.collectAllMembers instead")
  def getIntermediateAndLeafs(d: Data): Seq[Data] = collectAllMembers(d)

  /** Recursively collect just the leaf components of a data component's children
    * (i.e. anything that isn't a `Record` or a `Vec`, but an `Element`).
    * Probes of aggregates are also considered leaves.
    *
    * @param d Data component to recursively collect leaf components.
    *
    * @return All `Element` components; intermediate fields/indices are not included
    */
  def collectLeafMembers(d: Data): Seq[Data] =
    DataMirror
      .collectMembers(d) {
        case x: Element => x
        case x if hasProbeTypeModifier(x) => x
      }
      .toVector

  /** Recursively collect all expanded member components of a data component, including
    * intermediate aggregate nodes
    *
    * @param d Data component to recursively collect components.
    *
    * @return All member components; intermediate fields/indices ARE included
    */
  def collectAllMembers(d: Data): Seq[Data] = collectMembers(d) { case x => x }.toVector

  /** Recursively collects all fields selected by collector within a data and additionally generates
    * path names for each field
    * Accepts a collector partial function, rather than a collector function
    *
    * @param data Data to collect fields, as well as all children datas it directly and indirectly instantiates
    * @param path Recursively generated path name, starting with a root path
    * @param collector Collector partial function to pick which components to collect
    *
    * @return A sequence of pairs that map a data field to its corresponding path name
    *
    * @tparam T Type of the component that will be collected
    */
  private[chisel3] def collectMembersAndPaths[T](
    d:         Data,
    path:      String = ""
  )(collector: PartialFunction[Data, T]
  ): Iterable[(T, String)] = new Iterable[(T, String)] {
    def iterator = {
      val myItems = collector.lift(d).map { x => (x -> path) }
      val deepChildrenItems = d match {
        case a: Record if (!hasProbeTypeModifier(a)) =>
          a._elements.iterator.flatMap {
            case (fieldName, fieldData) =>
              collectMembersAndPaths(fieldData, s"$path.$fieldName")(collector)
          }
        case a: Vec[_] if (!hasProbeTypeModifier(a)) =>
          a.elementsIterator.zipWithIndex.flatMap {
            case (fieldData, fieldIndex) =>
              collectMembersAndPaths(fieldData, s"$path($fieldIndex)")(collector)
          }
        case other => Nil
      }
      myItems.iterator ++ deepChildrenItems
    }
  }

  /** Collects all fields selected by collector within a data and all recursive children fields
    * Accepts a collector partial function, rather than a collector function
    *
    * @param data Data to collect fields, as well as all children datas it directly and indirectly instantiates
    * @param collector Collector partial function to pick which components to collect
    * @tparam T Type of the component that will be collected
    */
  def collectMembers[T](d: Data)(collector: PartialFunction[Data, T]): Iterable[T] = new Iterable[T] {
    def iterator = {
      val myItems = collector.lift(d)
      val deepChildrenItems = d match {
        case a: Aggregate if (!hasProbeTypeModifier(a)) =>
          a.elementsIterator.flatMap { x => collectMembers(x)(collector) }
        case other => Nil
      }
      myItems.iterator ++ deepChildrenItems
    }
  }

  // Alignment-aware collections
  import connectable.{AlignedWithRoot, Alignment, ConnectableAlignment, FlippedWithRoot}
  // Implement typeclass to enable collecting over Alignment
  implicit val AlignmentMatchingZipOfChildren: HasMatchingZipOfChildren[Alignment] =
    new HasMatchingZipOfChildren[Alignment] {
      def matchingZipOfChildren(
        left:  Option[Alignment],
        right: Option[Alignment]
      ): Seq[(Option[Alignment], Option[Alignment])] =
        Alignment.matchingZipOfChildren(left, right)
    }

  // Implement typeclass to enable collecting over ConnectableAlignment
  implicit val ConnectableAlignmentMatchingZipOfChildren: HasMatchingZipOfChildren[ConnectableAlignment] =
    new HasMatchingZipOfChildren[ConnectableAlignment] {
      def matchingZipOfChildren(
        left:  Option[ConnectableAlignment],
        right: Option[ConnectableAlignment]
      ): Seq[(Option[ConnectableAlignment], Option[ConnectableAlignment])] =
        ConnectableAlignment.matchingZipOfChildren(left, right)
    }

  /** Collects all members of base who are aligned w.r.t. base
    * Accepts a collector partial function, rather than a collector function
    *
    * @param base Data from whom aligned members (w.r.t. base) are collected
    * @param collector Collector partial function to pick which components to collect
    * @tparam T Type of the component that will be collected
    */
  def collectAlignedDeep[T](base: Data)(pf: PartialFunction[Data, T]): Seq[T] = {
    collectMembersOverAllForAny(Some(Alignment(base, true)), None) {
      case (Some(x: AlignedWithRoot), _) => (pf.lift(x.member), None)
    }.map(_._1).flatten
  }

  /** Collects all members of base who are flipped w.r.t. base
    * Accepts a collector partial function, rather than a collector function
    *
    * @param base Data from whom flipped members (w.r.t. base) are collected
    * @param collector Collector partial function to pick which components to collect
    * @tparam T Type of the component that will be collected
    */
  def collectFlippedDeep[T](base: Data)(pf: PartialFunction[Data, T]): Seq[T] = {
    collectMembersOverAllForAny(Some(Alignment(base, true)), None) {
      case (Some(x: FlippedWithRoot), _) => (pf.lift(x.member), None)
    }.map(_._1).flatten
  }

  /** Typeclass trait to use collectMembersOverMatches, collectMembersOverAll, collectMembersOverAllForAny, collectMembersOverAllForAnyFunction */
  trait HasMatchingZipOfChildren[T] {
    def matchingZipOfChildren(left: Option[T], right: Option[T]): Seq[(Option[T], Option[T])]
  }

  /** Collects over members left and right who have structurally corresponding members in both left and right
    * Accepts a collector partial function, rather than a collector function
    *
    * @param left Data from whom members are collected
    * @param right Data from whom members are collected
    * @param collector Collector partial function to pick which components from left and right to collect
    * @tparam T Type of the thing being collected
    */
  def collectMembersOverMatches[D: HasMatchingZipOfChildren, T](
    left:      D,
    right:     D
  )(collector: PartialFunction[(D, D), T]
  ): Seq[T] = {
    def newCollector(lOpt: Option[D], rOpt: Option[D]): Option[(Option[T], Option[Unit])] = {
      (lOpt, rOpt) match {
        case (Some(l), Some(r)) =>
          collector.lift((l, r)) match {
            case Some(x) => Some((Some(x), None))
            case None    => None
          }
        case other => None
      }
    }
    collectMembersOverAllForAnyFunction(Some(left), Some(right)) {
      case (Some(l), Some(r)) =>
        collector.lift((l, r)) match {
          case Some(x) => Some((Some(x), None))
          case None    => None
        }
      case other => None
    }.collect {
      case (Some(x), None) => (x)
    }
  }

  /** Collects over members left and right who have structurally corresponding members in either left and right
    * Accepts a collector partial function, rather than a collector function
    *
    * @param left Data from whom members are collected
    * @param right Data from whom members are collected
    * @param collector Collector partial function to pick which components from left, right, or both to collect
    * @tparam T Type of the thing being collected
    */
  def collectMembersOverAll[D: HasMatchingZipOfChildren, T](
    left:      D,
    right:     D
  )(collector: PartialFunction[(Option[D], Option[D]), T]
  ): Seq[T] = {
    collectMembersOverAllForAnyFunction(Some(left), Some(right)) {
      case (lOpt: Option[D], rOpt: Option[D]) =>
        collector.lift((lOpt, rOpt)) match {
          case Some(x) => Some((Some(x), None))
          case None    => None
        }
    }.collect {
      case (Some(x), None) => x
    }
  }

  /** Collects over members left and right who have structurally corresponding members in either left and right
    * Can return an optional value for left, right, both or neither
    * Accepts a collector partial function, rather than a collector function
    *
    * @param left Data from whom members are collected
    * @param right Data from whom members are collected
    * @param collector Collector partial function to pick which components from left, right, or both to collect
    * @tparam L Type of the thing being collected from the left
    * @tparam R Type of the thing being collected from the right
    */
  def collectMembersOverAllForAny[D: HasMatchingZipOfChildren, L, R](
    left:       Option[D],
    right:      Option[D]
  )(pcollector: PartialFunction[(Option[D], Option[D]), (Option[L], Option[R])]
  ): Seq[(Option[L], Option[R])] = {
    collectMembersOverAllForAnyFunction(left, right)(pcollector.lift)
  }

  /** Collects over members left and right who have structurally corresponding members in either left and right
    * Can return an optional value for left, right, both or neither
    * Accepts a full function
    *
    * @param left Data from whom members are collected
    * @param right Data from whom members are collected
    * @param collector Collector full function to pick which components from left, right, or both to collect
    * @tparam L Type of the thing being collected from the left
    * @tparam R Type of the thing being collected from the right
    */
  def collectMembersOverAllForAnyFunction[D: HasMatchingZipOfChildren, L, R](
    left:      Option[D],
    right:     Option[D]
  )(collector: ((Option[D], Option[D])) => Option[(Option[L], Option[R])]
  ): Seq[(Option[L], Option[R])] = {
    val myItems = collector((left, right)) match {
      case None               => Nil
      case Some((None, None)) => Nil
      case Some(other)        => Seq(other)
    }
    val matcher = implicitly[HasMatchingZipOfChildren[D]]
    val childItems = matcher.matchingZipOfChildren(left, right).flatMap {
      case (l, r) => collectMembersOverAllForAnyFunction(l, r)(collector)
    }
    myItems ++ childItems
  }

  // Function to path upwards, stopping if reaching including
  private[chisel3] def modulePath(h: HasId, until: Option[BaseModule]): Seq[BaseModule] = {
    val me = h match {
      case m: BaseModule => Seq(m)
      case d: Data       => d.topBinding.location.toSeq
      case m: MemBase[_] => m._parent.toSeq
    }
    if (me == until.toSeq) Nil
    else {
      me ++ me.flatMap(x => x._parent.toSeq.flatMap(p => modulePath(p, until)))
    }
  }
  // Function to find the least common ancestor of two nodes
  private[chisel3] def leastCommonAncestorModule(left: HasId, right: HasId): Option[BaseModule] = {
    val leftPath = modulePath(left, None)
    val leftPathSet = leftPath.toSet
    val rightPath = modulePath(right, None)
    rightPath.collectFirst { case p if leftPathSet.contains(p) => p }
  }
  // Returns LCA paths if a common ancestor exists.  The returned paths includes the LCA.
  private[chisel3] def findLCAPaths(left: HasId, right: HasId): Option[(Seq[BaseModule], Seq[BaseModule])] = {
    leastCommonAncestorModule(left, right).map { lca =>
      (modulePath(left, Some(lca)) ++ Seq(lca), modulePath(right, Some(lca)) ++ Seq(lca))
    }
  }
}
