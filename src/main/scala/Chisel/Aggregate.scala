// See LICENSE for license details.

package Chisel
import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashSet, LinkedHashMap}
import Builder.pushCommand

/** An abstract class for data types that solely consist of (are an aggregate
  * of) other Data objects.
  */
sealed abstract class Aggregate(dirArg: Direction) extends Data(dirArg) {
  private[Chisel] def cloneTypeWidth(width: Width): this.type = cloneType
  def width: Width = flatten.map(_.width).reduce(_ + _)
}

object Vec {
  /** Creates a new [[Vec]] with `n` entries of the specified data type.
    *
    * @note elements are NOT assigned by default and have no value
    */
  def apply[T <: Data](n: Int, gen: T): Vec[T] = new Vec(gen.cloneType, n)

  @deprecated("Vec argument order should be size, t; this will be removed by the official release", "chisel3")
  def apply[T <: Data](gen: T, n: Int): Vec[T] = new Vec(gen.cloneType, n)

  /** Creates a new [[Vec]] composed of elements of the input Seq of [[Data]]
    * nodes.
    *
    * @note input elements should be of the same type
    * @note the width of all output elements is the width of the largest input
    * element
    * @note output elements are connected from the input elements
    */
  def apply[T <: Data](elts: Seq[T]): Vec[T] = {
    // REVIEW TODO: error checking to guard against type mismatch?

    require(!elts.isEmpty)
    val width = elts.map(_.width).reduce(_ max _)
    val vec = new Vec(elts.head.cloneTypeWidth(width), elts.length)
    pushCommand(DefWire(vec))
    for ((v, e) <- vec zip elts)
      v := e
    vec
  }

  /** Creates a new [[Vec]] composed of the input [[Data]] nodes.
    *
    * @note input elements should be of the same type
    * @note the width of all output elements is the width of the largest input
    * element
    * @note output elements are connected from the input elements
    */
  def apply[T <: Data](elt0: T, elts: T*): Vec[T] =
    // REVIEW TODO: does this really need to exist as a standard function?
    apply(elt0 +: elts.toSeq)

  /** Creates a new [[Vec]] of length `n` composed of the results of the given
    * function applied over a range of integer values starting from 0.
    *
    * @param n number of elements in the vector (the function is applied from
    * 0 to `n-1`)
    * @param gen function that takes in an Int (the index) and returns a
    * [[Data]] that becomes the output element
    */
  def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T] =
    apply((0 until n).map(i => gen(i)))

  /** Creates a new [[Vec]] of length `n` composed of the result of the given
    * function repeatedly applied.
    *
    * @param n number of elements (amd the number of times the function is
    * called)
    * @param gen function that generates the [[Data]] that becomes the output
    * element
    */
  def fill[T <: Data](n: Int)(gen: => T): Vec[T] = apply(Seq.fill(n)(gen))
}

/** A vector (array) of [[Data]] elements. Provides hardware versions of various
  * collection transformation functions found in software array implementations.
  *
  * @tparam T type of elements
  */
sealed class Vec[T <: Data] private (gen: => T, val length: Int)
    extends Aggregate(gen.dir) with VecLike[T] {
  // REVIEW TODO: should this take a Seq instead of a gen()?

  private val self = IndexedSeq.fill(length)(gen)

  override def <> (that: Data): Unit = that match {
    case _: Vec[_] => this bulkConnect that
    case _ => this badConnect that
  }

  def <> (that: Seq[T]): Unit =
    // REVIEW TODO: come up with common style: match on type in body or
    // multiple invocation signatures
    for ((a, b) <- this zip that)
      a <> b

  def <> (that: Vec[T]): Unit = this bulkConnect that
    // REVIEW TODO: standardize as above

  override def := (that: Data): Unit = that match {
    case _: Vec[_] => this connect that
    case _ => this badConnect that
  }

  def := (that: Seq[T]): Unit = {
    // REVIEW TODO: standardize as above
    require(this.length == that.length)
    for ((a, b) <- this zip that)
      a := b
  }

  def := (that: Vec[T]): Unit = this connect that

  /** Creates a dynamically indexed read accessor into the array. Generates
    * logic (likely some kind of multiplexer).
    */
  def apply(idx: UInt): T = {
    val x = gen
    // REVIEW TODO: what happens when people try to assign into this?
    // Should this be a read-only reference?
    pushCommand(DefAccessor(x, Node(this), NO_DIR, idx.ref))
    x
  }

  /** Creates a statically indexed read accessor into the array. Generates no
    * logic.
    */
  def apply(idx: Int): T = self(idx)

  def read(idx: UInt): T = apply(idx)
  // REVIEW TODO: does this need to exist?

  def write(idx: UInt, data: T): Unit = apply(idx) := data

  override def cloneType: this.type =
    Vec(gen, length).asInstanceOf[this.type]

  private val t = gen
  private[Chisel] def toType: String = s"${t.toType}[$length]"
  private[Chisel] lazy val flatten: IndexedSeq[Bits] =
    (0 until length).flatMap(i => this.apply(i).flatten)

  for ((elt, i) <- self zipWithIndex)
    elt.setRef(this, i)
}

/** A trait for [[Vec]]s containing common hardware generators for collection
  * operations.
  */
trait VecLike[T <: Data] extends collection.IndexedSeq[T] {
  def read(idx: UInt): T
  // REVIEW TODO: does this need to exist? (does the same thing as apply)

  def write(idx: UInt, data: T): Unit
  def apply(idx: UInt): T

  /** Outputs true if p outputs true for every element.
    *
    * This generates into a function evaluation followed by a logical AND
    * reduction.
    */
  def forall(p: T => Bool): Bool = (this map p).fold(Bool(true))(_ && _)

  /** Outputs true if p outputs true for at least one element.
    *
    * This generates into a function evaluation followed by a logical OR
    * reduction.
    */
  def exists(p: T => Bool): Bool = (this map p).fold(Bool(false))(_ || _)

  /** Outputs true if the vector contains at least one element equal to x (using
    * the === operator).
    *
    * This generates into an equality comparison followed by a logical OR
    * reduction.
    */
  def contains(x: T)(implicit evidence: T <:< UInt): Bool = this.exists(_ === x)

  /** Outputs the number of elements for which p is true.
    *
    * This generates into a function evaluation followed by a set bit counter.
    */
  def count(p: T => Bool): UInt = PopCount((this map p).toSeq)

  /** Helper function that appends an index (literal value) to each element,
    * useful for hardware generators which output an index.
    */
  private def indexWhereHelper(p: T => Bool) = this map p zip (0 until length).map(i => UInt(i))

  /** Outputs the index of the first element for which p outputs true.
    *
    * This generates into a function evaluation followed by a priority mux.
    */
  def indexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p))

  /** Outputs the index of the last element for which p outputs true.
    *
    * This generates into a function evaluation followed by a priority mux.
    */
  def lastIndexWhere(p: T => Bool): UInt = PriorityMux(indexWhereHelper(p).reverse)

  /** Outputs the index of the element for which p outputs true, assuming that
    * the there is exactly one such element.
    *
    * This generates into a function evaluation followed by a one-hot mux. The
    * implementation may be more efficient than a priority mux, but incorrect
    * results are possible if there is not exactly one true element.
    */
  def onlyIndexWhere(p: T => Bool): UInt = Mux1H(indexWhereHelper(p))
  // REVIEW TODO: can (should?) this be assertion checked?
}

/** Base class for data types defined as a bundle of other data types.
  *
  * Usage: extend this class (either as an anonymous or named class) and define
  * members variables of [[Data]] subtypes to be elements in the Bundle.
  */
class Bundle extends Aggregate(NO_DIR) {
  private val _namespace = Builder.globalNamespace.child

  // REVIEW TODO: perhaps deprecate to match FIRRTL semantics? Also needs
  // strong connect operator.
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
  override def <> (that: Data): Unit = that match {
    case _: Bundle => this bulkConnect that
    case _ => this badConnect that
  }

  // REVIEW TODO: should there be different semantics for this? Or just ban it?
  override def := (that: Data): Unit = this <> that

  lazy val elements: ListMap[String, Data] = ListMap(namedElts:_*)

  /** Returns a best guess at whether a field in this Bundle is a user-defined
    * Bundle element.
    */
  private def isBundleField(m: java.lang.reflect.Method) =
    m.getParameterTypes.isEmpty &&
    !java.lang.reflect.Modifier.isStatic(m.getModifiers) &&
    classOf[Data].isAssignableFrom(m.getReturnType) &&
    !(Bundle.keywords contains m.getName) && !(m.getName contains '$')

  /** Returns a list of elements in this Bundle.
    */
  private[Chisel] lazy val namedElts = {
    val nameMap = LinkedHashMap[String, Data]()
    val seen = HashSet[Data]()
    for (m <- getClass.getMethods.sortWith(_.getName < _.getName); if isBundleField(m)) {
      m.invoke(this) match {
        case d: Data =>
          if (nameMap contains m.getName) {
            require(nameMap(m.getName) eq d)
          } else if (!seen(d)) {
            nameMap(m.getName) = d; seen += d
          }
        case _ =>
      }
    }
    ArrayBuffer(nameMap.toSeq:_*) sortWith {case ((an, a), (bn, b)) => (a._id > b._id) || ((a eq b) && (an > bn))}
  }
  private[Chisel] def toType = {
    def eltPort(elt: Data): String = {
      val flipStr = if (elt.isFlip) "flip " else ""
      s"${flipStr}${elt.getRef.name} : ${elt.toType}"
    }
    s"{${namedElts.reverse.map(e => eltPort(e._2)).mkString(", ")}}"
  }
  private[Chisel] lazy val flatten = namedElts.flatMap(_._2.flatten)
  private[Chisel] def addElt(name: String, elt: Data): Unit =
    namedElts += name -> elt
  private[Chisel] override def _onModuleClose: Unit =
    for ((name, elt) <- namedElts) { elt.setRef(this, _namespace.name(name)) }

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
          case _: java.lang.reflect.InvocationTargetException =>
            Builder.error(s"Parameterized Bundle ${this.getClass} needs cloneType method. You are probably using " +
              "an anonymous Bundle object that captures external state and hence is un-cloneTypeable")
            this
        }
      case _: java.lang.reflect.InvocationTargetException | _: java.lang.IllegalArgumentException =>
        Builder.error(s"Parameterized Bundle ${this.getClass} needs cloneType method")
        this
    }
  }
}

object Bundle {
  private val keywords =
    HashSet[String]("flip", "asInput", "asOutput", "cloneType", "toBits")

  def apply[T <: Bundle](b: => T)(implicit p: Parameters): T = {
    Builder.paramsScope(p.push){ b }
  }

  //TODO @deprecated("Use Chisel.paramsScope object","08-01-2015")
  def apply[T <: Bundle](b: => T,  f: PartialFunction[Any,Any]): T = {
    val q = Builder.getParams.alterPartial(f)
    apply(b)(q)
  }
}
