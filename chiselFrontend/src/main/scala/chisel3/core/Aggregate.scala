// See LICENSE for license details.

package chisel3.core

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashSet, LinkedHashMap}
import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._

/** An abstract class for data types that solely consist of (are an aggregate
  * of) other Data objects.
  */
sealed abstract class Aggregate extends Data {
  private[core] def cloneTypeWidth(width: Width): this.type = cloneType
  private[core] def width: Width = flatten.map(_.width).reduce(_ + _)
  private[core] def legacyConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit =
    pushCommand(BulkConnect(sourceInfo, this.lref, that.lref))

  override def do_asUInt(implicit sourceInfo: SourceInfo): UInt = SeqUtils.do_asUInt(this.flatten)
}

object Vec {
  /** Creates a new [[Vec]] with `n` entries of the specified data type.
    *
    * @note elements are NOT assigned by default and have no value
    */
  def apply[T <: Data](n: Int, gen: T)(implicit compileOptions: CompileOptions): Vec[T] =
    new Vec(gen.chiselCloneType, n)

  @deprecated("Vec argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](gen: T, n: Int)(implicit compileOptions: CompileOptions): Vec[T] =
    apply(n, gen)

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

  def do_apply[T <: Data](elts: Seq[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] = {
    // REVIEW TODO: this should be removed in favor of the apply(elts: T*)
    // varargs constructor, which is more in line with the style of the Scala
    // collection API. However, a deprecation phase isn't possible, since
    // changing apply(elt0, elts*) to apply(elts*) causes a function collision
    // with apply(Seq) after type erasure. Workarounds by either introducing a
    // DummyImplicit or additional type parameter will break some code.

    // Check that types are homogeneous.  Width mismatch for Elements is safe.
    require(!elts.isEmpty)
    def eltsCompatible(a: Data, b: Data) = a match {
      case _: Element => a.getClass == b.getClass
      case _: Aggregate => Mux.typesCompatible(a, b)
    }

    val t = elts.head
    for (e <- elts.tail)
      require(eltsCompatible(t, e), s"can't create Vec of heterogeneous types ${t.getClass} and ${e.getClass}")

    val maxWidth = elts.map(_.width).reduce(_ max _)
    val vec = Wire(new Vec(t.cloneTypeWidth(maxWidth).chiselCloneType, elts.length))
    def doConnect(sink: T, source: T) = {
      if (elts.head.flatten.exists(_.dir != Direction.Unspecified)) {
        sink bulkConnect source
      } else {
        sink connect source
      }
    }
    for ((v, e) <- vec zip elts) {
      doConnect(v, e)
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

  def do_tabulate[T <: Data](n: Int)(gen: (Int) => T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
    apply((0 until n).map(i => gen(i)))

  /** Creates a new [[Vec]] of length `n` composed of the result of the given
    * function repeatedly applied.
    *
    * @param n number of elements (amd the number of times the function is
    * called)
    * @param gen function that generates the [[Data]] that becomes the output
    * element
    */
  @deprecated("Vec.fill(n)(gen) is deprecated. Please use Vec(Seq.fill(n)(gen))", "chisel3")
  def fill[T <: Data](n: Int)(gen: => T): Vec[T] = macro VecTransform.fill

  def do_fill[T <: Data](n: Int)(gen: => T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
    apply(Seq.fill(n)(gen))

  /** Truncate an index to implement modulo-power-of-2 addressing. */
  private[core] def truncateIndex(idx: UInt, n: Int)(implicit sourceInfo: SourceInfo): UInt = {
    val w = BigInt(n-1).bitLength
    if (n <= 1) UInt(0)
    else if (idx.width.known && idx.width.get <= w) idx
    else if (idx.width.known) idx(w-1,0)
    else (idx | UInt(0, w))(w-1,0)
  }
}

/** A vector (array) of [[Data]] elements. Provides hardware versions of various
  * collection transformation functions found in software array implementations.
  *
  * @tparam T type of elements
  * @note when multiple conflicting assignments are performed on a Vec element,
  * the last one takes effect (unlike Mem, where the result is undefined)
  * @note Vecs, unlike classes in Scala's collection library, are propagated
  * intact to FIRRTL as a vector type, which may make debugging easier
  */
sealed class Vec[T <: Data] private (gen: => T, val length: Int)
    extends Aggregate with VecLike[T] {
  // Note: the constructor takes a gen() function instead of a Seq to enforce
  // that all elements must be the same and because it makes FIRRTL generation
  // simpler.
  private val self: Seq[T] = Vector.fill(length)(gen)

  /**
  * sample_element 'tracks' all changes to the elements of self.
  * For consistency, sample_element is always used for creating dynamically
  * indexed ports and outputing the FIRRTL type.
  *
  * Needed specifically for the case when the Vec is length 0.
  */
  private[core] val sample_element: T = gen

  // allElements current includes sample_element
  // This is somewhat weird although I think the best course of action here is
  // to deprecate allElements in favor of dispatched functions to Data or
  // a pattern matched recursive descent
  private[chisel3] final def allElements: Seq[Element] =
    (sample_element +: self).flatMap(_.allElements)

  /** Strong bulk connect, assigning elements in this Vec from elements in a Seq.
    *
    * @note the length of this Vec must match the length of the input Seq
    */
  def <> (that: Seq[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit = {
    require(this.length == that.length)
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
  def apply(idx: UInt): T = {
    Binding.checkSynthesizable(idx ,s"'idx' ($idx)")
    val port = gen
    val i = Vec.truncateIndex(idx, length)(UnlocatableSourceInfo)
    port.setRef(this, i)

    // Bind each element of port to being whatever the base type is
    // Using the head element as the sample_element
    for((port_elem, model_elem) <- port.allElements zip sample_element.allElements) {
      port_elem.binding = model_elem.binding
    }

    port
  }

  /** Creates a statically indexed read or write accessor into the array.
    */
  def apply(idx: Int): T = self(idx)

  @deprecated("Use Vec.apply instead", "chisel3")
  def read(idx: UInt): T = apply(idx)

  @deprecated("Use Vec.apply instead", "chisel3")
  def write(idx: UInt, data: T): Unit = {
    apply(idx).:=(data)(DeprecatedSourceInfo, chisel3.core.ExplicitCompileOptions.NotStrict)
  }

  override def cloneType: this.type = {
    new Vec(gen, length).asInstanceOf[this.type]
  }

  private[chisel3] def toType: String = s"${sample_element.toType}[$length]"
  private[chisel3] lazy val flatten: IndexedSeq[Bits] =
    (0 until length).flatMap(i => this.apply(i).flatten)

  for ((elt, i) <- self zipWithIndex)
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

/** A trait for [[Vec]]s containing common hardware generators for collection
  * operations.
  */
trait VecLike[T <: Data] extends collection.IndexedSeq[T] with HasId {
  def apply(idx: UInt): T

  // IndexedSeq has its own hashCode/equals that we must not use
  override def hashCode: Int = super[HasId].hashCode
  override def equals(that: Any): Boolean = super[HasId].equals(that)
  
  @deprecated("Use Vec.apply instead", "chisel3")
  def read(idx: UInt): T

  @deprecated("Use Vec.apply instead", "chisel3")
  def write(idx: UInt, data: T): Unit

  /** Outputs true if p outputs true for every element.
    */
  def forall(p: T => Bool): Bool = macro SourceInfoTransform.pArg

  def do_forall(p: T => Bool)(implicit sourceInfo: SourceInfo): Bool =
    (this map p).fold(Bool(true))(_ && _)

  /** Outputs true if p outputs true for at least one element.
    */
  def exists(p: T => Bool): Bool = macro SourceInfoTransform.pArg

  def do_exists(p: T => Bool)(implicit sourceInfo: SourceInfo): Bool =
    (this map p).fold(Bool(false))(_ || _)

  /** Outputs true if the vector contains at least one element equal to x (using
    * the === operator).
    */
  def contains(x: T)(implicit ev: T <:< UInt): Bool = macro VecTransform.contains

  def do_contains(x: T)(implicit sourceInfo: SourceInfo, ev: T <:< UInt): Bool =
    this.exists(_ === x)

  /** Outputs the number of elements for which p is true.
    */
  def count(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  def do_count(p: T => Bool)(implicit sourceInfo: SourceInfo): UInt =
    SeqUtils.count(this map p)

  /** Helper function that appends an index (literal value) to each element,
    * useful for hardware generators which output an index.
    */
  private def indexWhereHelper(p: T => Bool) = this map p zip (0 until length).map(i => UInt(i))

  /** Outputs the index of the first element for which p outputs true.
    */
  def indexWhere(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  def do_indexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo): UInt =
    SeqUtils.priorityMux(indexWhereHelper(p))

  /** Outputs the index of the last element for which p outputs true.
    */
  def lastIndexWhere(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  def do_lastIndexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo): UInt =
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
  def onlyIndexWhere(p: T => Bool): UInt = macro CompileOptionsTransform.pArg

  def do_onlyIndexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    SeqUtils.oneHotMux(indexWhereHelper(p))
}

/** Base class for data types defined as a bundle of other data types.
  *
  * Usage: extend this class (either as an anonymous or named class) and define
  * members variables of [[Data]] subtypes to be elements in the Bundle.
  */
class Bundle extends Aggregate {
  private val _namespace = Builder.globalNamespace.child

  // TODO: replace with better defined FIRRTL weak-connect operator
  /** Connect elements in this Bundle to elements in `that` on a best-effort
    * (weak) basis, matching by type, orientation, and name.
    *
    * @note unconnected elements will NOT generate errors or warnings
    *
    * @example
    * {{{
    * // Pass through wires in this module's io to those mySubModule's io,
    * // matching by type, orientation, and name, and ignoring extra wires.
    * mySubModule.io <> io
    * }}}
    */

  lazy val elements: ListMap[String, Data] = ListMap(namedElts:_*)

  /** Returns a best guess at whether a field in this Bundle is a user-defined
    * Bundle element without looking at type signatures.
    */
  private def isBundleField(m: java.lang.reflect.Method) =
    m.getParameterTypes.isEmpty &&
    !java.lang.reflect.Modifier.isStatic(m.getModifiers) &&
    !(Bundle.keywords contains m.getName) && !(m.getName contains '$')

  /** Returns a field's contained user-defined Bundle element if it appears to
    * be one, otherwise returns None.
    */
  private def getBundleField(m: java.lang.reflect.Method): Option[Data] = m.invoke(this) match {
    case d: Data => Some(d)
    case Some(d: Data) => Some(d)
    case _ => None
  }

  /** Returns a list of elements in this Bundle.
    */
  private[core] lazy val namedElts = {
    val nameMap = LinkedHashMap[String, Data]()
    val seen = HashSet[Data]()
    for (m <- getPublicFields(classOf[Bundle])) {
      getBundleField(m) foreach { d =>
        if (nameMap contains m.getName) {
          require(nameMap(m.getName) eq d)
        } else if (!seen(d)) {
          nameMap(m.getName) = d
          seen += d
        }
      }
    }
    ArrayBuffer(nameMap.toSeq:_*) sortWith {case ((an, a), (bn, b)) => (a._id > b._id) || ((a eq b) && (an > bn))}
  }
  private[chisel3] def toType = {
    def eltPort(elt: Data): String = {
      val flipStr: String = if(Data.isFirrtlFlipped(elt)) "flip " else ""
      s"${flipStr}${elt.getRef.name} : ${elt.toType}"
    }
    s"{${namedElts.reverse.map(e => eltPort(e._2)).mkString(", ")}}"
  }
  private[chisel3] lazy val flatten = namedElts.flatMap(_._2.flatten)
  private[chisel3] override def _onModuleClose: Unit = // scalastyle:ignore method.name
    for ((name, elt) <- namedElts) { elt.setRef(this, _namespace.name(name)) }
    
  private[chisel3] final def allElements: Seq[Element] = namedElts.flatMap(_._2.allElements)

  override def cloneType : this.type = {
    // If the user did not provide a cloneType method, try invoking one of
    // the following constructors, not all of which necessarily exist:
    // - A zero-parameter constructor
    // - A one-paramater constructor, with null as the argument
    // - A one-parameter constructor for a nested Bundle, with the enclosing
    //   parent Module as the argument
    val constructor = this.getClass.getConstructors.head
    try {
      val args = Seq.fill(constructor.getParameterTypes.size)(null)
      constructor.newInstance(args:_*).asInstanceOf[this.type]
    } catch {
      case e: java.lang.reflect.InvocationTargetException if e.getCause.isInstanceOf[java.lang.NullPointerException] =>
        try {
          constructor.newInstance(_parent.get).asInstanceOf[this.type]
        } catch {
          case _: java.lang.reflect.InvocationTargetException | _: java.lang.IllegalArgumentException =>
            Builder.error(s"Parameterized Bundle ${this.getClass} needs cloneType method. You are probably using " +
              "an anonymous Bundle object that captures external state and hence is un-cloneTypeable")
            this
        }
      case _: java.lang.reflect.InvocationTargetException | _: java.lang.IllegalArgumentException =>
        Builder.error(s"Parameterized Bundle ${this.getClass} needs cloneType method")
        this
    }
  }

  /** Default "pretty-print" implementation
    * Analogous to printing a Map
    * Results in "Bundle(elt0.name -> elt0.value, ...)"
    */
  def toPrintable: Printable = {
    val elts =
      if (elements.isEmpty) List.empty[Printable]
      else {
        elements.toList.reverse flatMap { case (name, data) =>
          List(PString(s"$name -> "), data.toPrintable, PString(", "))
        } dropRight 1 // Remove trailing ", "
      }
    PString("Bundle(") + Printables(elts) + PString(")")
  }
}

private[core] object Bundle {
  val keywords = List("flip", "asInput", "asOutput", "cloneType", "chiselCloneType", "toBits",
    "widthOption", "signalName", "signalPathName", "signalParent", "signalComponent")
}
