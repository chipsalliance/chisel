// See LICENSE for license details.

package chisel3.core

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashSet, LinkedHashMap}
import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._
import chisel3.SourceInfoDoc

/** An abstract class for data types that solely consist of (are an aggregate
  * of) other Data objects.
  */
sealed abstract class Aggregate extends Data {
  private[chisel3] override def bind(target: Binding, parentDirection: SpecifiedDirection) {
    binding = target

    val resolvedDirection = SpecifiedDirection.fromParent(parentDirection, specifiedDirection)
    for (child <- getElements) {
      child.bind(ChildBinding(this), resolvedDirection)
    }

    // Check that children obey the directionality rules.
    val childDirections = getElements.map(_.direction).toSet
    direction = if (childDirections == Set()) {  // Sadly, Scala can't do set matching
      // If empty, use my assigned direction
      resolvedDirection match {
        case SpecifiedDirection.Unspecified | SpecifiedDirection.Flip => ActualDirection.Unspecified
        case SpecifiedDirection.Output => ActualDirection.Output
        case SpecifiedDirection.Input => ActualDirection.Input
      }
    } else if (childDirections == Set(ActualDirection.Unspecified)) {
      ActualDirection.Unspecified
    } else if (childDirections == Set(ActualDirection.Input)) {
      ActualDirection.Input
    } else if (childDirections == Set(ActualDirection.Output)) {
      ActualDirection.Output
    } else if (childDirections subsetOf
        Set(ActualDirection.Output, ActualDirection.Input,
            ActualDirection.Bidirectional(ActualDirection.Default),
            ActualDirection.Bidirectional(ActualDirection.Flipped))) {
      resolvedDirection match {
        case SpecifiedDirection.Unspecified => ActualDirection.Bidirectional(ActualDirection.Default)
        case SpecifiedDirection.Flip => ActualDirection.Bidirectional(ActualDirection.Flipped)
        case _ => throw new RuntimeException("Unexpected forced Input / Output")
      }
    } else {
      this match {
        // Anything flies in compatibility mode
        case t: Record if !t.compileOptions.dontAssumeDirectionality => resolvedDirection match {
          case SpecifiedDirection.Unspecified => ActualDirection.Bidirectional(ActualDirection.Default)
          case SpecifiedDirection.Flip => ActualDirection.Bidirectional(ActualDirection.Flipped)
          case _ => ActualDirection.Bidirectional(ActualDirection.Default)
        }
        case _ =>
          val childWithDirections = getElements zip getElements.map(_.direction)
          throw Binding.MixedDirectionAggregateException(s"Aggregate '$this' can't have elements that are both directioned and undirectioned: $childWithDirections")
      }
    }
  }

  override def litOption: Option[BigInt] = ???  // TODO implement me

  // Returns the LitArg of a Bits object.
  // Internal API for Bundle literals, to copy the LitArg of argument literals into the top map.
  protected def litArgOfBits(elt: Bits): LitArg = elt.litArgOption.get

  /** Returns a Seq of the immediate contents of this Aggregate, in order.
    */
  def getElements: Seq[Data]

  private[chisel3] def width: Width = getElements.map(_.width).foldLeft(0.W)(_ + _)
  private[core] def legacyConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit = {
    // If the source is a DontCare, generate a DefInvalid for the sink,
    //  otherwise, issue a Connect.
    if (that == DontCare) {
      pushCommand(DefInvalid(sourceInfo, Node(this)))
    } else {
      pushCommand(BulkConnect(sourceInfo, Node(this), Node(that)))
    }
  }

  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    SeqUtils.do_asUInt(flatten.map(_.asUInt()))
  }
  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    var i = 0
    val bits = WireDefault(UInt(this.width), that)  // handles width padding
    for (x <- flatten) {
      x.connectFromBits(bits(i + x.getWidth - 1, i))
      i += x.getWidth
    }
  }
}

trait VecFactory extends SourceInfoDoc {
  /** Creates a new [[Vec]] with `n` entries of the specified data type.
    *
    * @note elements are NOT assigned by default and have no value
    */
  def apply[T <: Data](n: Int, gen: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] = {
    if (compileOptions.declaredTypeMustBeUnbound) {
      requireIsChiselType(gen, "vec type")
    }
    new Vec(gen.cloneTypeFull, n)
  }

  /** Truncate an index to implement modulo-power-of-2 addressing. */
  private[core] def truncateIndex(idx: UInt, n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    val w = BigInt(n-1).bitLength
    if (n <= 1) 0.U
    else if (idx.width.known && idx.width.get <= w) idx
    else if (idx.width.known) idx(w-1,0)
    else (idx | 0.U(w.W))(w-1,0)
  }
}

object Vec extends VecFactory

/** A vector (array) of [[Data]] elements. Provides hardware versions of various
  * collection transformation functions found in software array implementations.
  *
  * Careful consideration should be given over the use of [[Vec]] vs
  * [[scala.collection.immutable.Seq Seq]] or some other Scala collection. In general [[Vec]] only
  * needs to be used when there is a need to express the hardware collection in a [[Reg]] or IO
  * [[Bundle]] or when access to elements of the array is indexed via a hardware signal.
  *
  * Example of indexing into a [[Vec]] using a hardware address and where the [[Vec]] is defined in
  * an IO [[Bundle]]
  *
  *  {{{
  *    val io = IO(new Bundle {
  *      val in = Input(Vec(20, UInt(16.W)))
  *      val addr = UInt(5.W)
  *      val out = Output(UInt(16.W))
  *    })
  *    io.out := io.in(io.addr)
  *  }}}
  *
  * @tparam T type of elements
  *
  * @note
  *  - when multiple conflicting assignments are performed on a Vec element, the last one takes effect (unlike Mem, where the result is undefined)
  *  - Vecs, unlike classes in Scala's collection library, are propagated intact to FIRRTL as a vector type, which may make debugging easier
  */
sealed class Vec[T <: Data] private[core] (gen: => T, val length: Int)
    extends Aggregate with VecLike[T] {
  override def toString: String = {
    s"$sample_element[$length]$bindingToString"
  }

  private[core] override def typeEquivalent(that: Data): Boolean = that match {
    case that: Vec[T] =>
      this.length == that.length &&
      (this.sample_element typeEquivalent that.sample_element)
    case _ => false
  }

  // Note: the constructor takes a gen() function instead of a Seq to enforce
  // that all elements must be the same and because it makes FIRRTL generation
  // simpler.
  private val self: Seq[T] = Vector.fill(length)(gen)

  /**
  * sample_element 'tracks' all changes to the elements.
  * For consistency, sample_element is always used for creating dynamically
  * indexed ports and outputing the FIRRTL type.
  *
  * Needed specifically for the case when the Vec is length 0.
  */
  private[chisel3] val sample_element: T = gen

  // allElements current includes sample_element
  // This is somewhat weird although I think the best course of action here is
  // to deprecate allElements in favor of dispatched functions to Data or
  // a pattern matched recursive descent
  private[chisel3] final override def allElements: Seq[Element] =
    (sample_element +: self).flatMap(_.allElements)

  /** Strong bulk connect, assigning elements in this Vec from elements in a Seq.
    *
    * @note the length of this Vec must match the length of the input Seq
    */
  def <> (that: Seq[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit = {
    if (this.length != that.length) {
      Builder.error("Vec and Seq being bulk connected have different lengths!")
    }
    for ((a, b) <- this zip that)
      a <> b
  }

  // TODO: eliminate once assign(Seq) isn't ambiguous with assign(Data) since Vec extends Seq and Data
  def <> (that: Vec[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit = this bulkConnect that.asInstanceOf[Data]

  /** Strong bulk connect, assigning elements in this Vec from elements in a Seq.
    *
    * @note the length of this Vec must match the length of the input Seq
    */
  def := (that: Seq[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit = {
    require(this.length == that.length)
    for ((a, b) <- this zip that)
      a := b
  }

  // TODO: eliminate once assign(Seq) isn't ambiguous with assign(Data) since Vec extends Seq and Data
  def := (that: Vec[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit = this connect that

  /** Creates a dynamically indexed read or write accessor into the array.
    */
  override def apply(p: UInt): T = macro CompileOptionsTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_apply(p: UInt)(implicit compileOptions: CompileOptions): T = {
    requireIsHardware(p, "vec index")
    val port = gen

    // Reconstruct the resolvedDirection (in Aggregate.bind), since it's not stored.
    // It may not be exactly equal to that value, but the results are the same.
    val reconstructedResolvedDirection = direction match {
      case ActualDirection.Input => SpecifiedDirection.Input
      case ActualDirection.Output => SpecifiedDirection.Output
      case ActualDirection.Bidirectional(ActualDirection.Default) | ActualDirection.Unspecified =>
        SpecifiedDirection.Unspecified
      case ActualDirection.Bidirectional(ActualDirection.Flipped) => SpecifiedDirection.Flip
    }
    // TODO port technically isn't directly child of this data structure, but the result of some
    // muxes / demuxes. However, this does make access consistent with the top-level bindings.
    // Perhaps there's a cleaner way of accomplishing this...
    port.bind(ChildBinding(this), reconstructedResolvedDirection)

    val i = Vec.truncateIndex(p, length)(UnlocatableSourceInfo, compileOptions)
    port.setRef(this, i)

    port
  }

  /** Creates a statically indexed read or write accessor into the array.
    */
  def apply(idx: Int): T = self(idx)

  override def cloneType: this.type = {
    new Vec(gen.cloneTypeFull, length).asInstanceOf[this.type]
  }

  override def getElements: Seq[Data] =
    (0 until length).map(apply(_))

  for ((elt, i) <- self.zipWithIndex)
    elt.setRef(this, i)

  /** Default "pretty-print" implementation
    * Analogous to printing a Seq
    * Results in "Vec(elt0, elt1, ...)"
    */
  def toPrintable: Printable = {
    val elts =
      if (length == 0) List.empty[Printable]
      else self flatMap (e => List(e.toPrintable, PString(", "))) dropRight 1
    PString("Vec(") + Printables(elts) + PString(")")
  }
}

object VecInit extends SourceInfoDoc {
  /** Creates a new [[Vec]] composed of elements of the input Seq of [[Data]]
    * nodes.
    *
    * @note input elements should be of the same type (this is checked at the
    * FIRRTL level, but not at the Scala / Chisel level)
    * @note the width of all output elements is the width of the largest input
    * element
    * @note output elements are connected from the input elements
    */
  def apply[T <: Data](elts: Seq[T]): Vec[T] = macro VecTransform.apply_elts

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](elts: Seq[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] = {
    // REVIEW TODO: this should be removed in favor of the apply(elts: T*)
    // varargs constructor, which is more in line with the style of the Scala
    // collection API. However, a deprecation phase isn't possible, since
    // changing apply(elt0, elts*) to apply(elts*) causes a function collision
    // with apply(Seq) after type erasure. Workarounds by either introducing a
    // DummyImplicit or additional type parameter will break some code.

    // Check that types are homogeneous.  Width mismatch for Elements is safe.
    require(!elts.isEmpty)
    elts.foreach(requireIsHardware(_, "vec element"))

    val vec = Wire(Vec(elts.length, cloneSupertype(elts, "Vec")))

    // TODO: try to remove the logic for this mess
    elts.head.direction match {
      case ActualDirection.Input | ActualDirection.Output | ActualDirection.Unspecified =>
        // When internal wires are involved, driver / sink must be specified explicitly, otherwise
        // the system is unable to infer which is driver / sink
        (vec zip elts).foreach(x => x._1 := x._2)
      case ActualDirection.Bidirectional(_) =>
        // For bidirectional, must issue a bulk connect so subelements are resolved correctly.
        // Bulk connecting two wires may not succeed because Chisel frontend does not infer
        // directions.
        (vec zip elts).foreach(x => x._1 <> x._2)
    }
    vec
  }

  /** Creates a new [[Vec]] composed of the input [[Data]] nodes.
    *
    * @note input elements should be of the same type (this is checked at the
    * FIRRTL level, but not at the Scala / Chisel level)
    * @note the width of all output elements is the width of the largest input
    * element
    * @note output elements are connected from the input elements
    */
  def apply[T <: Data](elt0: T, elts: T*): Vec[T] = macro VecTransform.apply_elt0

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](elt0: T, elts: T*)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
    apply(elt0 +: elts.toSeq)

  /** Creates a new [[Vec]] of length `n` composed of the results of the given
    * function applied over a range of integer values starting from 0.
    *
    * @param n number of elements in the vector (the function is applied from
    * 0 to `n-1`)
    * @param gen function that takes in an Int (the index) and returns a
    * [[Data]] that becomes the output element
    */
  def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T] = macro VecTransform.tabulate

  /** @group SourceInfoTransformMacro */
  def do_tabulate[T <: Data](n: Int)(gen: (Int) => T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
    apply((0 until n).map(i => gen(i)))
}

/** A trait for [[Vec]]s containing common hardware generators for collection
  * operations.
  */
trait VecLike[T <: Data] extends collection.IndexedSeq[T] with HasId with SourceInfoDoc {
  def apply(p: UInt): T = macro CompileOptionsTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_apply(p: UInt)(implicit compileOptions: CompileOptions): T

  // IndexedSeq has its own hashCode/equals that we must not use
  override def hashCode: Int = super[HasId].hashCode
  override def equals(that: Any): Boolean = super[HasId].equals(that)

  @chiselRuntimeDeprecated
  @deprecated("Use Vec.apply instead", "chisel3")
  def read(idx: UInt)(implicit compileOptions: CompileOptions): T = do_apply(idx)(compileOptions)

  @chiselRuntimeDeprecated
  @deprecated("Use Vec.apply instead", "chisel3")
  def write(idx: UInt, data: T)(implicit compileOptions: CompileOptions): Unit = {
    do_apply(idx)(compileOptions).:=(data)(DeprecatedSourceInfo, compileOptions)
  }

  /** Outputs true if p outputs true for every element.
    */
  def forall(p: T => Bool): Bool = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_forall(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    (this map p).fold(true.B)(_ && _)

  /** Outputs true if p outputs true for at least one element.
    */
  def exists(p: T => Bool): Bool = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_exists(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    (this map p).fold(false.B)(_ || _)

  /** Outputs true if the vector contains at least one element equal to x (using
    * the === operator).
    */
  def contains(x: T)(implicit ev: T <:< UInt): Bool = macro VecTransform.contains

  /** @group SourceInfoTransformMacro */
  def do_contains(x: T)(implicit sourceInfo: SourceInfo, ev: T <:< UInt, compileOptions: CompileOptions): Bool =
    this.exists(_ === x)

  /** Outputs the number of elements for which p is true.
    */
  def count(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_count(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    SeqUtils.count(this map p)

  /** Helper function that appends an index (literal value) to each element,
    * useful for hardware generators which output an index.
    */
  private def indexWhereHelper(p: T => Bool) = this map p zip (0 until length).map(i => i.asUInt)

  /** Outputs the index of the first element for which p outputs true.
    */
  def indexWhere(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_indexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    SeqUtils.priorityMux(indexWhereHelper(p))

  /** Outputs the index of the last element for which p outputs true.
    */
  def lastIndexWhere(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_lastIndexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    SeqUtils.priorityMux(indexWhereHelper(p).reverse)

  /** Outputs the index of the element for which p outputs true, assuming that
    * the there is exactly one such element.
    *
    * The implementation may be more efficient than a priority mux, but
    * incorrect results are possible if there is not exactly one true element.
    *
    * @note the assumption that there is only one element for which p outputs
    * true is NOT checked (useful in cases where the condition doesn't always
    * hold, but the results are not used in those cases)
    */
  def onlyIndexWhere(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_onlyIndexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    SeqUtils.oneHotMux(indexWhereHelper(p))
}

/** Base class for Aggregates based on key values pairs of String and Data
  *
  * Record should only be extended by libraries and fairly sophisticated generators.
  * RTL writers should use [[Bundle]].  See [[Record#elements]] for an example.
  */
abstract class Record(private[chisel3] implicit val compileOptions: CompileOptions) extends Aggregate {

  /** The collection of [[Data]]
    *
    * This underlying datastructure is a ListMap because the elements must
    * remain ordered for serialization/deserialization. Elements added later
    * are higher order when serialized (this is similar to [[Vec]]). For example:
    * {{{
    *   // Assume we have some type MyRecord that creates a Record from the ListMap
    *   val record = MyRecord(ListMap("fizz" -> UInt(16.W), "buzz" -> UInt(16.W)))
    *   // "buzz" is higher order because it was added later than "fizz"
    *   record("fizz") := "hdead".U
    *   record("buzz") := "hbeef".U
    *   val uint = record.asUInt
    *   assert(uint === "hbeefdead".U) // This will pass
    * }}}
    */
  override def toString: String = {
    val bindingString = topBindingOpt match {
      case Some(BundleLitBinding(_)) =>
        val contents = elements.toList.reverse.map { case (name, data) =>
          s"$name=$data"
        }.mkString(", ")
        s"($contents)"
      case _ => bindingToString
    }
    s"$className$bindingString"
  }

  val elements: ListMap[String, Data]

  /** Name for Pretty Printing */
  def className: String = this.getClass.getSimpleName

  private[core] override def typeEquivalent(that: Data): Boolean = that match {
    case that: Record =>
      this.getClass == that.getClass &&
      this.elements.size == that.elements.size &&
      this.elements.forall{case (name, model) =>
        that.elements.contains(name) &&
        (that.elements(name) typeEquivalent model)}
    case _ => false
  }

  // NOTE: This sets up dependent references, it can be done before closing the Module
  private[chisel3] override def _onModuleClose: Unit = { // scalastyle:ignore method.name
    // Since Bundle names this via reflection, it is impossible for two elements to have the same
    // identifier; however, Namespace sanitizes identifiers to make them legal for Firrtl/Verilog
    // which can cause collisions
    val _namespace = Namespace.empty
    for ((name, elt) <- elements) { elt.setRef(this, _namespace.name(name, leadingDigitOk=true)) }
  }

  private[chisel3] final def allElements: Seq[Element] = elements.toIndexedSeq.flatMap(_._2.allElements)

  override def getElements: Seq[Data] = elements.toIndexedSeq.map(_._2)

  // Helper because Bundle elements are reversed before printing
  private[chisel3] def toPrintableHelper(elts: Seq[(String, Data)]): Printable = {
    val xs =
      if (elts.isEmpty) List.empty[Printable] // special case because of dropRight below
      else elts flatMap { case (name, data) =>
             List(PString(s"$name -> "), data.toPrintable, PString(", "))
           } dropRight 1 // Remove trailing ", "
    PString(s"$className(") + Printables(xs) + PString(")")
  }
  /** Default "pretty-print" implementation
    * Analogous to printing a Map
    * Results in "`\$className(elt0.name -> elt0.value, ...)`"
    */
  def toPrintable: Printable = toPrintableHelper(elements.toList)
}

/**
  * Mix-in for Bundles that have arbitrary Seqs of Chisel types that aren't
  * involved in hardware construction.
  *
  * Used to avoid raising an error/exception when a Seq is a public member of the
  * bundle.
  * This is useful if we those public Seq fields in the Bundle are unrelated to
  * hardware construction.
  */
trait IgnoreSeqInBundle {
  this: Bundle =>

  override def ignoreSeq: Boolean = true
}

class AutoClonetypeException(message: String) extends ChiselException(message)

/** Base class for data types defined as a bundle of other data types.
  *
  * Usage: extend this class (either as an anonymous or named class) and define
  * members variables of [[Data]] subtypes to be elements in the Bundle.
  *
  * Example of an anonymous IO bundle
  * {{{
  *   class MyModule extends Module {
  *     val io = IO(new Bundle {
  *       val in = Input(UInt(64.W))
  *       val out = Output(SInt(128.W))
  *     })
  *   }
  * }}}
  *
  * Or as a named class
  * {{{
  *   class Packet extends Bundle {
  *     val header = UInt(16.W)
  *     val addr   = UInt(16.W)
  *     val data   = UInt(32.W)
  *   }
  *   class MyModule extends Module {
  *      val io = IO(new Bundle {
  *        val inPacket = Input(new Packet)
  *        val outPacket = Output(new Packet)
  *      })
  *      val reg = Reg(new Packet)
  *      reg <> inPacket
  *      outPacket <> reg
  *   }
  * }}}
  */
abstract class Bundle(implicit compileOptions: CompileOptions) extends Record {
  override def className: String = this.getClass.getSimpleName match {
    case name if name.startsWith("$anon$") => "AnonymousBundle"  // fallback for anonymous Bundle case
    case "" => "AnonymousBundle"  // ditto, but on other platforms
    case name => name
  }
  /** The collection of [[Data]]
    *
    * Elements defined earlier in the Bundle are higher order upon
    * serialization. For example:
    * @example
    * {{{
    *   class MyBundle extends Bundle {
    *     val foo = UInt(16.W)
    *     val bar = UInt(16.W)
    *   }
    *   // Note that foo is higher order because its defined earlier in the Bundle
    *   val bundle = Wire(new MyBundle)
    *   bundle.foo := 0x1234.U
    *   bundle.bar := 0x5678.U
    *   val uint = bundle.asUInt
    *   assert(uint === "h12345678".U) // This will pass
    * }}}
    */
  final lazy val elements: ListMap[String, Data] = {
    val nameMap = LinkedHashMap[String, Data]()
    val seen = HashSet[Data]()
    for (m <- getPublicFields(classOf[Bundle])) {
      getBundleField(m) match {
        case Some(d: Data) =>
          if (nameMap contains m.getName) {
            require(nameMap(m.getName) eq d)
          } else if (!seen(d)) {
            nameMap(m.getName) = d
            seen += d
          }
        case None =>
          if (!ignoreSeq) {
            m.invoke(this) match {
              case s: scala.collection.Seq[Any] if s.nonEmpty => s.head match {
                // Ignore empty Seq()
                case d: Data => throwException("Public Seq members cannot be used to define Bundle elements " +
                  s"(found public Seq member '${m.getName}'). " +
                  "Either use a Vec if all elements are of the same type, or MixedVec if the elements " +
                  "are of different types. If this Seq member is not intended to construct RTL, mix in the trait " +
                  "IgnoreSeqInBundle.")
                case _ => // don't care about non-Data Seq
              }
              case _ => // not a Seq
            }
          }
      }
    }
    ListMap(nameMap.toSeq sortWith { case ((an, a), (bn, b)) => (a._id > b._id) || ((a eq b) && (an > bn)) }: _*)
  }

  /**
    * Overridden by [[IgnoreSeqInBundle]] to allow arbitrary Seqs of Chisel elements.
    */
  def ignoreSeq: Boolean = false

  /** Returns a field's contained user-defined Bundle element if it appears to
    * be one, otherwise returns None.
    */
  private def getBundleField(m: java.lang.reflect.Method): Option[Data] = m.invoke(this) match {
    case d: Data => Some(d)
    case Some(d: Data) => Some(d)
    case _ => None
  }

  // Memoize the outer instance for autoclonetype, especially where this is context-dependent
  // (like the outer module or enclosing Bundles).
  private var _outerInst: Option[Object] = None

  // For autoclonetype, record possible candidates for outer instance.
  // _outerInst should always take precedence, since it should be propagated from the original
  // object which has the most accurate context.
  private val _containingModule: Option[BaseModule] = Builder.currentModule
  private val _containingBundles: Seq[Bundle] = Builder.updateBundleStack(this)

  override def cloneType : this.type = {
    // This attempts to infer constructor and arguments to clone this Bundle subtype without
    // requiring the user explicitly overriding cloneType.
    import scala.language.existentials
    import scala.reflect.runtime.universe._

    val clazz = this.getClass

    def autoClonetypeError(desc: String): Nothing = {
      throw new AutoClonetypeException(s"Unable to automatically infer cloneType on $clazz: $desc")
    }

    val mirror = runtimeMirror(clazz.getClassLoader)
    val classSymbolOption = try {
      Some(mirror.reflect(this).symbol)
    } catch {
      case e: scala.reflect.internal.Symbols#CyclicReference => None  // Workaround for a scala bug
    }

    val enclosingClassOption = (clazz.getEnclosingClass, classSymbolOption) match {
      case (null, _) => None
      case (_, Some(classSymbol)) if classSymbol.isStatic => None  // allows support for members of companion objects
      case (outerClass, _) => Some(outerClass)
    }

    // For compatibility with pre-3.1, where null is tried as an argument to the constructor.
    // This stores potential error messages which may be used later.
    var outerClassError: Option[String] = None

    // Check if this is an inner class, and if so, try to get the outer instance
    val outerClassInstance = enclosingClassOption.map { outerClass =>
      def canAssignOuterClass(x: Object) = outerClass.isAssignableFrom(x.getClass)

      val outerInstance = _outerInst match {
        case Some(outerInstance) => outerInstance  // use _outerInst if defined
        case None =>  // determine outer instance if not already recorded
          try {
            // Prefer this if it works, but doesn't work in all cases, namely anonymous inner Bundles
            val outer = clazz.getDeclaredField("$outer").get(this)
            _outerInst = Some(outer)
            outer
          } catch {
            case (_: NoSuchFieldException | _: IllegalAccessException) =>
              // Fallback using guesses based on common patterns
              val allOuterCandidates = Seq(
                  _containingModule.toSeq,
                  _containingBundles
                ).flatten.distinct
              allOuterCandidates.filter(canAssignOuterClass(_)) match {
                case outer :: Nil =>
                  _outerInst = Some(outer)  // record the guess for future use
                  outer
                case Nil =>  // TODO: replace with fatal autoClonetypeError once compatibility period is dropped
                  outerClassError = Some(s"Unable to determine instance of outer class $outerClass," +
                      s" no candidates assignable to outer class types; examined $allOuterCandidates")
                  null
                case candidates => // TODO: replace with fatal autoClonetypeError once compatibility period is dropped
                  outerClassError = Some(s"Unable to determine instance of outer class $outerClass," +
                      s" multiple possible candidates $candidates assignable to outer class type")
                  null
              }
          }
      }
      (outerClass, outerInstance)
    }

    // If possible (constructor with no arguments), try Java reflection first
    // This handles two cases that Scala reflection doesn't:
    // 1. getting the ClassSymbol of a class with an anonymous outer class fails with a
    //    CyclicReference exception
    // 2. invoking the constructor of an anonymous inner class seems broken (it expects the outer
    //    class as an argument, but fails because the number of arguments passed in is incorrect)
    if (clazz.getConstructors.size == 1) {
      var ctor = clazz.getConstructors.head
      val argTypes = ctor.getParameterTypes.toList
      val clone = (argTypes, outerClassInstance) match {
        case (Nil, None) => // no arguments, no outer class, invoke constructor directly
          Some(ctor.newInstance().asInstanceOf[this.type])
        case (argType :: Nil, Some((_, outerInstance))) =>
          if (outerInstance == null) {
            Builder.deprecated(s"chisel3.1 autoclonetype failed, falling back to 3.0 behavior using null as the outer instance." +
                s" Autoclonetype failure reason: ${outerClassError.get}",
                Some(s"$clazz"))
            Some(ctor.newInstance(outerInstance).asInstanceOf[this.type])
          } else if (argType isAssignableFrom outerInstance.getClass) {
            Some(ctor.newInstance(outerInstance).asInstanceOf[this.type])
          } else {
            None
          }
        case _ => None

      }
      clone match {
        case Some(clone) =>
          clone._outerInst = this._outerInst
          if (!clone.typeEquivalent(this)) {
            autoClonetypeError(s"automatically cloned $clone not type-equivalent to base." +
            " Constructor argument values were not inferred, ensure constructor is deterministic.")
          }
          return clone.asInstanceOf[this.type]
        case None =>
      }
    }

    // Get constructor parameters and accessible fields
    val classSymbol = classSymbolOption.getOrElse(autoClonetypeError(s"scala reflection failed." +
        " This is known to occur with inner classes on anonymous outer classes." +
        " In those cases, autoclonetype only works with no-argument constructors, or you can define a custom cloneType."))

    val decls = classSymbol.typeSignature.decls
    val ctors = decls.collect { case meth: MethodSymbol if meth.isConstructor => meth }
    if (ctors.size != 1) {
      autoClonetypeError(s"found multiple constructors ($ctors)." +
          " Either remove all but the default constructor, or define a custom cloneType method.")
    }
    val ctor = ctors.head
    val ctorParamss = ctor.paramLists
    val ctorParams = ctorParamss match {
      case Nil => List()
      case ctorParams :: Nil => ctorParams
      case ctorParams :: ctorImplicits :: Nil => ctorParams ++ ctorImplicits
      case _ => autoClonetypeError(s"internal error, unexpected ctorParamss = $ctorParamss")
    }
    val ctorParamsNames = ctorParams.map(_.name.toString)

    // Special case for anonymous inner classes: their constructor consists of just the outer class reference
    // Scala reflection on anonymous inner class constructors seems broken
    if (ctorParams.size == 1 && outerClassInstance.isDefined &&
        ctorParams.head.typeSignature == mirror.classSymbol(outerClassInstance.get._1).toType) {
      // Fall back onto Java reflection
      val ctors = clazz.getConstructors
      require(ctors.size == 1)  // should be consistent with Scala constructors
      try {
        val clone = ctors.head.newInstance(outerClassInstance.get._2).asInstanceOf[this.type]
        clone._outerInst = this._outerInst
        return clone
      } catch {
        case e @ (_: java.lang.reflect.InvocationTargetException | _: IllegalArgumentException) =>
          autoClonetypeError(s"unexpected failure at constructor invocation, got $e.")
      }
    }

    // Get all the class symbols up to (but not including) Bundle and get all the accessors.
    // (each ClassSymbol's decls only includes those declared in the class itself)
    val bundleClassSymbol = mirror.classSymbol(classOf[Bundle])
    val superClassSymbols = classSymbol.baseClasses.takeWhile(_ != bundleClassSymbol)
    val superClassDecls = superClassSymbols.map(_.typeSignature.decls).flatten
    val accessors = superClassDecls.collect { case meth: MethodSymbol if meth.isParamAccessor => meth }

    // Get constructor argument values
    // Check that all ctor params are immutable and accessible. Immutability is required to avoid
    // potential subtle bugs (like values changing after cloning).
    // This also generates better error messages (all missing elements shown at once) instead of
    // failing at the use site one at a time.
    val accessorsName = accessors.filter(_.isStable).map(_.name.toString)
    val paramsDiff = ctorParamsNames.toSet -- accessorsName.toSet
    if (!paramsDiff.isEmpty) {
      autoClonetypeError(s"constructor has parameters (${paramsDiff.toList.sorted.mkString(", ")}) that are not both immutable and accessible." +
          " Either make all parameters immutable and accessible (vals) so cloneType can be inferred, or define a custom cloneType method.")
    }

    // Get all the argument values
    val accessorsMap = accessors.map(accessor => accessor.name.toString -> accessor).toMap
    val instanceReflect = mirror.reflect(this)
    val ctorParamsNameVals = ctorParamsNames.map {
      paramName => paramName -> instanceReflect.reflectMethod(accessorsMap(paramName)).apply()
    }

    // Opportunistic sanity check: ensure any arguments of type Data is not bound
    // (which could lead to data conflicts, since it's likely the user didn't know to re-bind them).
    // This is not guaranteed to catch all cases (for example, Data in Tuples or Iterables).
    val boundDataParamNames = ctorParamsNameVals.collect {
      case (paramName, paramVal: Data) if paramVal.topBindingOpt.isDefined => paramName
    }
    if (boundDataParamNames.nonEmpty) {
      autoClonetypeError(s"constructor parameters (${boundDataParamNames.sorted.mkString(", ")}) have values that are hardware types, which is likely to cause subtle errors." +
          " Use chisel types instead: use the value before it is turned to a hardware type (with Wire(...), Reg(...), etc) or use chiselTypeOf(...) to extract the chisel type.")
    }

    // Clone unbound parameters in case they are being used as bundle fields.
    val ctorParamsVals = ctorParamsNameVals.map {
      case (_, paramVal: Data) => paramVal.cloneTypeFull
      case (_, paramVal) => paramVal
    }

    // Invoke ctor
    val classMirror = outerClassInstance match {
      case Some((_, null)) => autoClonetypeError(outerClassError.get)  // deals with the null hack for 3.0 compatibility
      case Some((_, outerInstance)) => mirror.reflect(outerInstance).reflectClass(classSymbol)
      case _ => mirror.reflectClass(classSymbol)
    }
    val clone = classMirror.reflectConstructor(ctor).apply(ctorParamsVals:_*).asInstanceOf[this.type]
    clone._outerInst = this._outerInst

    if (!clone.typeEquivalent(this)) {
      autoClonetypeError(s"Automatically cloned $clone not type-equivalent to base $this." +
          " Constructor argument values were inferred: ensure that variable names are consistent and have the same value throughout the constructor chain," +
          " and that the constructor is deterministic.")
    }

    clone
  }

  /** Default "pretty-print" implementation
    * Analogous to printing a Map
    * Results in "`Bundle(elt0.name -> elt0.value, ...)`"
    * @note The order is reversed from the order of elements in order to print
    *   the fields in the order they were defined
    */
  override def toPrintable: Printable = toPrintableHelper(elements.toList.reverse)
}
