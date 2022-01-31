// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.VecLiterals.AddVecLiteralConstructor
import chisel3.experimental.dataview.{isView, reifySingleData, InvalidViewException}

import scala.collection.immutable.{SeqMap, VectorMap}
import scala.collection.mutable.{HashSet, LinkedHashMap}
import scala.language.experimental.macros
import chisel3.experimental.{BaseModule, BundleLiteralException, ChiselEnum, EnumType, VecLiteralException}
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._

import scala.collection.mutable

class AliasedAggregateFieldException(message: String) extends ChiselException(message)

/** An abstract class for data types that solely consist of (are an aggregate
  * of) other Data objects.
  */
sealed abstract class Aggregate extends Data {
  private[chisel3] override def bind(target: Binding, parentDirection: SpecifiedDirection): Unit = {
    _parent.foreach(_.addId(this))
    binding = target

    val resolvedDirection = SpecifiedDirection.fromParent(parentDirection, specifiedDirection)
    val duplicates = getElements.groupBy(identity).collect { case (x, elts) if elts.size > 1 => x }
    if (!duplicates.isEmpty) {
      this match {
        case b: Record =>
          // show groups of names of fields with duplicate id's
          // The sorts make the displayed order of fields deterministic and matching the order of occurrence in the Bundle.
          // It's a bit convoluted but happens rarely and makes the error message easier to understand
          val dupNames = duplicates.toSeq
            .sortBy(_._id)
            .map { duplicate =>
              b.elements.collect { case x if x._2._id == duplicate._id => x }.toSeq
                .sortBy(_._2._id)
                .map(_._1)
                .reverse
                .mkString("(", ",", ")")
            }
            .mkString(",")
          throw new AliasedAggregateFieldException(
            s"${b.className} contains aliased fields named ${dupNames}"
          )
        case _ =>
          throw new AliasedAggregateFieldException(
            s"Aggregate ${this.getClass} contains aliased fields $duplicates ${duplicates.mkString(",")}"
          )
      }
    }
    for (child <- getElements) {
      child.bind(ChildBinding(this), resolvedDirection)
    }

    // Check that children obey the directionality rules.
    val childDirections = getElements.map(_.direction).toSet - ActualDirection.Empty
    direction = ActualDirection.fromChildren(childDirections, resolvedDirection) match {
      case Some(dir) => dir
      case None =>
        val childWithDirections = getElements.zip(getElements.map(_.direction))
        throw MixedDirectionAggregateException(
          s"Aggregate '$this' can't have elements that are both directioned and undirectioned: $childWithDirections"
        )
    }
  }

  /** Return an Aggregate's literal value if it is a literal, None otherwise.
    * If any element of the aggregate is not a literal with a defined width, the result isn't a literal.
    *
    * @return an Aggregate's literal value if it is a literal.
    */
  override def litOption: Option[BigInt] = {
    // Shift the accumulated value by our width and add in our component, masked by our width.
    def shiftAdd(accumulator: Option[BigInt], elt: Data): Option[BigInt] = {
      (accumulator, elt.litOption) match {
        case (Some(accumulator), Some(eltLit)) =>
          val width = elt.width.get
          val masked = ((BigInt(1) << width) - 1) & eltLit // also handles the negative case with two's complement
          Some((accumulator << width) + masked)
        case (None, _) => None
        case (_, None) => None
      }
    }

    topBindingOpt match {
      case Some(BundleLitBinding(_)) | Some(VecLitBinding(_)) =>
        getElements.reverse
          .foldLeft[Option[BigInt]](Some(BigInt(0)))(shiftAdd)
      case _ => None
    }
  }

  /** Returns a Seq of the immediate contents of this Aggregate, in order.
    */
  def getElements: Seq[Data]

  private[chisel3] def width: Width = getElements.map(_.width).foldLeft(0.W)(_ + _)

  private[chisel3] def legacyConnect(that: Data)(implicit sourceInfo: SourceInfo): Unit = {
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

  private[chisel3] override def connectFromBits(
    that: Bits
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Unit = {
    var i = 0
    val bits = if (that.isLit) that else WireDefault(UInt(this.width), that) // handles width padding
    for (x <- flatten) {
      val fieldWidth = x.getWidth
      if (fieldWidth > 0) {
        x.connectFromBits(bits(i + fieldWidth - 1, i))
        i += fieldWidth
      } else {
        // There's a zero-width field in this bundle.
        // Zero-width fields can't really be assigned to, but the frontend complains if there are uninitialized fields,
        // so we assign it to DontCare. We can't use connectFromBits() on DontCare, so use := instead.
        x := DontCare
      }
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
  private[chisel3] def truncateIndex(
    idx: UInt,
    n:   BigInt
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): UInt = {
    val w = (n - 1).bitLength
    if (n <= 1) 0.U
    else if (idx.width.known && idx.width.get <= w) idx
    else if (idx.width.known) idx(w - 1, 0)
    else (idx | 0.U(w.W))(w - 1, 0)
  }
}

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
  *      val addr = Input(UInt(5.W))
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
sealed class Vec[T <: Data] private[chisel3] (gen: => T, val length: Int) extends Aggregate with VecLike[T] {

  override def toString: String = {
    topBindingOpt match {
      case Some(VecLitBinding(vecLitBinding)) =>
        val contents = vecLitBinding.zipWithIndex.map {
          case ((data, lit), index) =>
            s"$index=$lit"
        }.mkString(", ")
        s"${sample_element.cloneType}[$length]($contents)"
      case _ => stringAccessor(s"${sample_element.cloneType}[$length]")
    }
  }

  private[chisel3] override def typeEquivalent(that: Data): Boolean = that match {
    case that: Vec[T] =>
      this.length == that.length &&
        (this.sample_element.typeEquivalent(that.sample_element))
    case _ => false
  }

  private[chisel3] override def bind(target: Binding, parentDirection: SpecifiedDirection): Unit = {
    _parent.foreach(_.addId(this))
    binding = target

    val resolvedDirection = SpecifiedDirection.fromParent(parentDirection, specifiedDirection)
    sample_element.bind(SampleElementBinding(this), resolvedDirection)
    for (child <- getElements) { // assume that all children are the same
      child.bind(ChildBinding(this), resolvedDirection)
    }

    // Since all children are the same, we can just use the sample_element rather than all children
    // .get is safe because None means mixed directions, we only pass 1 so that's not possible
    direction = ActualDirection.fromChildren(Set(sample_element.direction), resolvedDirection).get
  }

  // Note: the constructor takes a gen() function instead of a Seq to enforce
  // that all elements must be the same and because it makes FIRRTL generation
  // simpler.
  private lazy val self: Seq[T] = {
    val _self = Vector.fill(length)(gen)
    for ((elt, i) <- _self.zipWithIndex)
      elt.setRef(this, i)
    _self
  }

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
  def <>(that: Seq[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit = {
    if (this.length != that.length) {
      Builder.error("Vec and Seq being bulk connected have different lengths!")
    }
    for ((a, b) <- this.zip(that))
      a <> b
  }

  // TODO: eliminate once assign(Seq) isn't ambiguous with assign(Data) since Vec extends Seq and Data
  def <>(that: Vec[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit =
    this.bulkConnect(that.asInstanceOf[Data])

  /** Strong bulk connect, assigning elements in this Vec from elements in a Seq.
    *
    * @note the length of this Vec must match the length of the input Seq
    */
  def :=(that: Seq[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit = {
    require(
      this.length == that.length,
      s"Cannot assign to a Vec of length ${this.length} from a Seq of different length ${that.length}"
    )
    for ((a, b) <- this.zip(that))
      a := b
  }

  // TODO: eliminate once assign(Seq) isn't ambiguous with assign(Data) since Vec extends Seq and Data
  def :=(that: Vec[T])(implicit sourceInfo: SourceInfo, moduleCompileOptions: CompileOptions): Unit = this.connect(that)

  /** Creates a dynamically indexed read or write accessor into the array.
    */
  override def apply(p: UInt): T = macro CompileOptionsTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_apply(p: UInt)(implicit compileOptions: CompileOptions): T = {
    requireIsHardware(this, "vec")
    requireIsHardware(p, "vec index")

    // Special handling for views
    if (isView(this)) {
      reifySingleData(this) match {
        // Views complicate things a bit, but views that correspond exactly to an identical Vec can just forward the
        // dynamic indexing to the target Vec
        // In theory, we could still do this forwarding if the sample element were different by deriving a DataView
        case Some(target: Vec[T @unchecked])
            if this.length == target.length &&
              this.sample_element.typeEquivalent(target.sample_element) =>
          return target.do_apply(p)
        case _ => throw InvalidViewException("Dynamic indexing of Views is not yet supported")
      }
    }

    val port = gen

    // Reconstruct the resolvedDirection (in Aggregate.bind), since it's not stored.
    // It may not be exactly equal to that value, but the results are the same.
    val reconstructedResolvedDirection = direction match {
      case ActualDirection.Input  => SpecifiedDirection.Input
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

  /** Default "pretty-print" implementation
    * Analogous to printing a Seq
    * Results in "Vec(elt0, elt1, ...)"
    */
  def toPrintable: Printable = {
    val elts =
      if (length == 0) List.empty[Printable]
      else self.flatMap(e => List(e.toPrintable, PString(", "))).dropRight(1)
    PString("Vec(") + Printables(elts) + PString(")")
  }

  /** A reduce operation in a tree like structure instead of sequentially
    * @example An adder tree
    * {{{
    * val sumOut = inputNums.reduceTree((a: T, b: T) => (a + b))
    * }}}
    */
  def reduceTree(redOp: (T, T) => T): T = macro VecTransform.reduceTreeDefault

  /** A reduce operation in a tree like structure instead of sequentially
    * @example A pipelined adder tree
    * {{{
    * val sumOut = inputNums.reduceTree(
    *   (a: T, b: T) => RegNext(a + b),
    *   (a: T) => RegNext(a)
    * )
    * }}}
    */
  def reduceTree(redOp: (T, T) => T, layerOp: (T) => T): T = macro VecTransform.reduceTree

  def do_reduceTree(
    redOp:   (T, T) => T,
    layerOp: (T) => T = (x: T) => x
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): T = {
    require(!isEmpty, "Cannot apply reduction on a vec of size 0")
    var curLayer: Seq[T] = this
    while (curLayer.length > 1) {
      curLayer = curLayer.grouped(2).map(x => if (x.length == 1) layerOp(x(0)) else redOp(x(0), x(1))).toSeq
    }
    curLayer(0)
  }

  /** Creates a Vec literal of this type with specified values. this must be a chisel type.
    *
    * @param elementInitializers literal values, specified as a pair of the Vec field to the literal value.
    * The Vec field is specified as a function from an object of this type to the field.
    * Fields that aren't initialized to DontCare, and assignment to a wire will overwrite any
    * existing value with DontCare.
    * @return a Vec literal of this type with subelement values specified
    *
    * Vec(2, UInt(8.W)).Lit(
    *   1 -> 0x0A.U,
    *   2 -> 0x0B.U
    * )
    * }}}
    */
  private[chisel3] def _makeLit(
    elementInitializers: (Int, T)*
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): this.type = {

    def checkLiteralConstruction(): Unit = {
      val dupKeys = elementInitializers.map { x => x._1 }.groupBy(x => x).flatMap {
        case (k, v) =>
          if (v.length > 1) {
            Some(k, v.length)
          } else {
            None
          }
      }
      if (dupKeys.nonEmpty) {
        throw new VecLiteralException(
          s"VecLiteral: has duplicated indices ${dupKeys.map { case (k, n) => s"$k($n times)" }.mkString(",")}"
        )
      }

      val outOfRangeIndices = elementInitializers.map(_._1).filter { case index => index < 0 || index >= length }
      if (outOfRangeIndices.nonEmpty) {
        throw new VecLiteralException(
          s"VecLiteral: The following indices (${outOfRangeIndices.mkString(",")}) " +
            s"are less than zero or greater or equal to than Vec length"
        )
      }
      cloneSupertype(elementInitializers.map(_._2), s"Vec.Lit(...)")

      // look for literals of this vec that are wider than the vec's type
      val badLits = elementInitializers.flatMap {
        case (index, lit) =>
          (sample_element.width, lit.width) match {
            case (KnownWidth(m), KnownWidth(n)) =>
              if (m < n) Some(index -> lit) else None
            case (KnownWidth(_), _) =>
              None
            case (UnknownWidth(), _) =>
              None
            case _ =>
              Some(index -> lit)
          }
        case _ => None
      }
      if (badLits.nonEmpty) {
        throw new VecLiteralException(
          s"VecLiteral: Vec[$gen] has the following incorrectly typed or sized initializers: " +
            badLits.map { case (a, b) => s"$a -> $b" }.mkString(",")
        )
      }

    }

    requireIsChiselType(this, "vec literal constructor model")
    checkLiteralConstruction()

    val clone = cloneType
    val cloneFields = getRecursiveFields(clone, "(vec root)").toMap

    // Create the Vec literal binding from litArgs of arguments
    val vecLitLinkedMap = new mutable.LinkedHashMap[Data, LitArg]()
    elementInitializers.sortBy { case (a, _) => a }.foreach {
      case (fieldIndex, value) =>
        val field = clone.apply(fieldIndex)
        val fieldName = cloneFields.getOrElse(
          field,
          throw new VecLiteralException(
            s"field $field (with value $value) is not a field," +
              s" ensure the field is specified as a function returning a field on an object of class ${this.getClass}," +
              s" eg '_.a' to select hypothetical bundle field 'a'"
          )
        )

        val valueBinding = value.topBindingOpt match {
          case Some(litBinding: LitBinding) => litBinding
          case _ => throw new VecLiteralException(s"field $fieldIndex specified with non-literal value $value")
        }

        field match { // Get the litArg(s) for this field
          case bitField: Bits =>
            if (!field.typeEquivalent(bitField)) {
              throw new VecLiteralException(
                s"VecLit: Literal specified at index $fieldIndex ($value) does not match Vec type $sample_element"
              )
            }
            if (bitField.getWidth > field.getWidth) {
              throw new VecLiteralException(
                s"VecLit: Literal specified at index $fieldIndex ($value) is too wide for Vec type $sample_element"
              )
            }
            val litArg = valueBinding match {
              case ElementLitBinding(litArg) => litArg
              case BundleLitBinding(litMap) =>
                litMap.getOrElse(
                  value,
                  throw new BundleLiteralException(s"Field $fieldName specified with unspecified value")
                )
              case VecLitBinding(litMap) =>
                litMap.getOrElse(
                  value,
                  throw new VecLiteralException(s"Field $fieldIndex specified with unspecified value")
                )
            }
            val adjustedLitArg = litArg.cloneWithWidth(sample_element.width)
            vecLitLinkedMap(bitField) = adjustedLitArg

          case recordField: Record =>
            if (!(recordField.typeEquivalent(value))) {
              throw new VecLiteralException(
                s"field $fieldIndex $recordField specified with non-type-equivalent value $value"
              )
            }
            // Copy the source BundleLitBinding with fields (keys) remapped to the clone
            val remap = getMatchedFields(value, recordField).toMap
            valueBinding.asInstanceOf[BundleLitBinding].litMap.map {
              case (valueField, valueValue) =>
                vecLitLinkedMap(remap(valueField)) = valueValue
            }

          case vecField: Vec[_] =>
            if (!(vecField.typeEquivalent(value))) {
              throw new VecLiteralException(
                s"field $fieldIndex $vecField specified with non-type-equivalent value $value"
              )
            }
            // Copy the source VecLitBinding with vecFields (keys) remapped to the clone
            val remap = getMatchedFields(value, vecField).toMap
            value.topBinding.asInstanceOf[VecLitBinding].litMap.map {
              case (valueField, valueValue) =>
                vecLitLinkedMap(remap(valueField)) = valueValue
            }

          case enumField: EnumType => {
            if (!(enumField.typeEquivalent(value))) {
              throw new VecLiteralException(
                s"field $fieldIndex $enumField specified with non-type-equivalent enum value $value"
              )
            }
            val litArg = valueBinding match {
              case ElementLitBinding(litArg) => litArg
              case _ =>
                throw new VecLiteralException(s"field $fieldIndex $enumField could not bematched with $valueBinding")
            }
            vecLitLinkedMap(field) = litArg
          }

          case _ => throw new VecLiteralException(s"unsupported field $fieldIndex of type $field")
        }
    }

    clone.bind(VecLitBinding(VectorMap(vecLitLinkedMap.toSeq: _*)))
    clone
  }
}

object VecInit extends SourceInfoDoc {

  /** Gets the correct connect operation (directed hardware assign or bulk connect) for element in Vec.
    */
  private def getConnectOpFromDirectionality[T <: Data](
    proto: T
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): (T, T) => Unit = proto.direction match {
    case ActualDirection.Input | ActualDirection.Output | ActualDirection.Unspecified =>
      // When internal wires are involved, driver / sink must be specified explicitly, otherwise
      // the system is unable to infer which is driver / sink
      (x, y) => x := y
    case ActualDirection.Bidirectional(_) =>
      // For bidirectional, must issue a bulk connect so subelements are resolved correctly.
      // Bulk connecting two wires may not succeed because Chisel frontend does not infer
      // directions.
      (x, y) => x <> y
  }

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
    require(elts.nonEmpty, "Vec hardware values are not allowed to be empty")
    elts.foreach(requireIsHardware(_, "vec element"))

    val vec = Wire(Vec(elts.length, cloneSupertype(elts, "Vec")))
    val op = getConnectOpFromDirectionality(vec.head)

    (vec.zip(elts)).foreach { x =>
      op(x._1, x._2)
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
  def do_tabulate[T <: Data](
    n:   Int
  )(gen: (Int) => T
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Vec[T] =
    apply((0 until n).map(i => gen(i)))

  /** Creates a new 2D [[Vec]] of length `n by m` composed of the results of the given
    * function applied over a range of integer values starting from 0.
    *
    * @param n number of 1D vectors inside outer vector
    * @param m number of elements in each 1D vector (the function is applied from
    * 0 to `n-1`)
    * @param gen function that takes in an Int (the index) and returns a
    * [[Data]] that becomes the output element
    */
  def tabulate[T <: Data](n: Int, m: Int)(gen: (Int, Int) => T): Vec[Vec[T]] = macro VecTransform.tabulate2D

  /** @group SourceInfoTransformMacro */
  def do_tabulate[T <: Data](
    n:   Int,
    m:   Int
  )(gen: (Int, Int) => T
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Vec[Vec[T]] = {
    // TODO make this lazy (requires LazyList and cross compilation, beyond the scope of this PR)
    val elts = Seq.tabulate(n, m)(gen)
    val flatElts = elts.flatten

    require(flatElts.nonEmpty, "Vec hardware values are not allowed to be empty")
    flatElts.foreach(requireIsHardware(_, "vec element"))

    val tpe = cloneSupertype(flatElts, "Vec.tabulate")
    val myVec = Wire(Vec(n, Vec(m, tpe)))
    val op = getConnectOpFromDirectionality(myVec.head.head)
    for {
      (xs1D, ys1D) <- myVec.zip(elts)
      (x, y) <- xs1D.zip(ys1D)
    } {
      op(x, y)
    }
    myVec
  }

  /** Creates a new 3D [[Vec]] of length `n by m by p` composed of the results of the given
    * function applied over a range of integer values starting from 0.
    *
    * @param n number of 2D vectors inside outer vector
    * @param m number of 1D vectors in each 2D vector
    * @param p number of elements in each 1D vector
    * @param gen function that takes in an Int (the index) and returns a
    * [[Data]] that becomes the output element
    */
  def tabulate[T <: Data](n: Int, m: Int, p: Int)(gen: (Int, Int, Int) => T): Vec[Vec[Vec[T]]] =
    macro VecTransform.tabulate3D

  /** @group SourceInfoTransformMacro */
  def do_tabulate[T <: Data](
    n:   Int,
    m:   Int,
    p:   Int
  )(gen: (Int, Int, Int) => T
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Vec[Vec[Vec[T]]] = {
    // TODO make this lazy (requires LazyList and cross compilation, beyond the scope of this PR)
    val elts = Seq.tabulate(n, m, p)(gen)
    val flatElts = elts.flatten.flatten

    require(flatElts.nonEmpty, "Vec hardware values are not allowed to be empty")
    flatElts.foreach(requireIsHardware(_, "vec element"))

    val tpe = cloneSupertype(flatElts, "Vec.tabulate")
    val myVec = Wire(Vec(n, Vec(m, Vec(p, tpe))))
    val op = getConnectOpFromDirectionality(myVec.head.head.head)

    for {
      (xs2D, ys2D) <- myVec.zip(elts)
      (xs1D, ys1D) <- xs2D.zip(ys2D)
      (x, y) <- xs1D.zip(ys1D)
    } {
      op(x, y)
    }

    myVec
  }

  /** Creates a new [[Vec]] of length `n` composed of the result of the given
    * function applied to an element of data type T.
    *
    * @param n number of elements in the vector
    * @param gen function that takes in an element T and returns an output
    * element of the same type
    */
  def fill[T <: Data](n: Int)(gen: => T): Vec[T] = macro VecTransform.fill

  /** @group SourceInfoTransformMacro */
  def do_fill[T <: Data](n: Int)(gen: => T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
    apply(Seq.fill(n)(gen))

  /** Creates a new 2D [[Vec]] of length `n by m` composed of the result of the given
    * function applied to an element of data type T.
    *
    * @param n number of inner vectors (rows) in the outer vector
    * @param m number of elements in each inner vector (column)
    * @param gen function that takes in an element T and returns an output
    * element of the same type
    */
  def fill[T <: Data](n: Int, m: Int)(gen: => T): Vec[Vec[T]] = macro VecTransform.fill2D

  /** @group SourceInfoTransformMacro */
  def do_fill[T <: Data](
    n:   Int,
    m:   Int
  )(gen: => T
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Vec[Vec[T]] = {
    do_tabulate(n, m)((_, _) => gen)
  }

  /** Creates a new 3D [[Vec]] of length `n by m by p` composed of the result of the given
    * function applied to an element of data type T.
    *
    * @param n number of 2D vectors inside outer vector
    * @param m number of 1D vectors in each 2D vector
    * @param p number of elements in each 1D vector
    * @param gen function that takes in an element T and returns an output
    * element of the same type
    */
  def fill[T <: Data](n: Int, m: Int, p: Int)(gen: => T): Vec[Vec[Vec[T]]] = macro VecTransform.fill3D

  /** @group SourceInfoTransformMacro */
  def do_fill[T <: Data](
    n:   Int,
    m:   Int,
    p:   Int
  )(gen: => T
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Vec[Vec[Vec[T]]] = {
    do_tabulate(n, m, p)((_, _, _) => gen)
  }

  /** Creates a new [[Vec]] of length `n` composed of the result of the given
    * function applied to an element of data type T.
    *
    * @param start First element in the Vec
    * @param len Lenth of elements in the Vec
    * @param f Function that applies the element T from previous index and returns the output
    * element to the next index
    */
  def iterate[T <: Data](start: T, len: Int)(f: (T) => T): Vec[T] = macro VecTransform.iterate

  /** @group SourceInfoTransformMacro */
  def do_iterate[T <: Data](
    start: T,
    len:   Int
  )(f:     (T) => T
  )(
    implicit sourceInfo: SourceInfo,
    compileOptions:      CompileOptions
  ): Vec[T] =
    apply(Seq.iterate(start, len)(f))
}

/** A trait for [[Vec]]s containing common hardware generators for collection
  * operations.
  */
trait VecLike[T <: Data] extends IndexedSeq[T] with HasId with SourceInfoDoc {
  def apply(p: UInt): T = macro CompileOptionsTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_apply(p: UInt)(implicit compileOptions: CompileOptions): T

  // IndexedSeq has its own hashCode/equals that we must not use
  override def hashCode: Int = super[HasId].hashCode
  override def equals(that: Any): Boolean = super[HasId].equals(that)

  /** Outputs true if p outputs true for every element.
    */
  def forall(p: T => Bool): Bool = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_forall(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    (this.map(p)).fold(true.B)(_ && _)

  /** Outputs true if p outputs true for at least one element.
    */
  def exists(p: T => Bool): Bool = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_exists(p: T => Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    (this.map(p)).fold(false.B)(_ || _)

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
    SeqUtils.count(this.map(p))

  /** Helper function that appends an index (literal value) to each element,
    * useful for hardware generators which output an index.
    */
  private def indexWhereHelper(p: T => Bool) = this.map(p).zip((0 until length).map(i => i.asUInt))

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

  // Doing this earlier than onModuleClose allows field names to be available for prefixing the names
  // of hardware created when connecting to one of these elements
  private def setElementRefs(): Unit = {
    // Since elements is a map, it is impossible for two elements to have the same
    // identifier; however, Namespace sanitizes identifiers to make them legal for Firrtl/Verilog
    // which can cause collisions
    val _namespace = Namespace.empty
    for ((name, elt) <- elements) { elt.setRef(this, _namespace.name(name, leadingDigitOk = true)) }
  }

  private[chisel3] override def bind(target: Binding, parentDirection: SpecifiedDirection): Unit = {
    try {
      super.bind(target, parentDirection)
    } catch { // nasty compatibility mode shim, where anything flies
      case e: MixedDirectionAggregateException if !compileOptions.dontAssumeDirectionality =>
        val resolvedDirection = SpecifiedDirection.fromParent(parentDirection, specifiedDirection)
        direction = resolvedDirection match {
          case SpecifiedDirection.Unspecified => ActualDirection.Bidirectional(ActualDirection.Default)
          case SpecifiedDirection.Flip        => ActualDirection.Bidirectional(ActualDirection.Flipped)
          case _                              => ActualDirection.Bidirectional(ActualDirection.Default)
        }
    }
    setElementRefs()
  }

  /** Creates a Bundle literal of this type with specified values. this must be a chisel type.
    *
    * @param elems literal values, specified as a pair of the Bundle field to the literal value.
    * The Bundle field is specified as a function from an object of this type to the field.
    * Fields that aren't initialized to DontCare, and assignment to a wire will overwrite any
    * existing value with DontCare.
    * @return a Bundle literal of this type with subelement values specified
    *
    * @example {{{
    * class MyBundle extends Bundle {
    *   val a = UInt(8.W)
    *   val b = Bool()
    * }
    *
    * (new MyBundle).Lit(
    *   _.a -> 42.U,
    *   _.b -> true.B
    * )
    * }}}
    */
  private[chisel3] def _makeLit(elems: (this.type => (Data, Data))*): this.type = {

    requireIsChiselType(this, "bundle literal constructor model")
    val clone = cloneType
    val cloneFields = getRecursiveFields(clone, "(bundle root)").toMap

    // Create the Bundle literal binding from litargs of arguments
    val bundleLitMap = elems.map { fn => fn(clone) }.flatMap {
      case (field, value) =>
        val fieldName = cloneFields.getOrElse(
          field,
          throw new BundleLiteralException(
            s"field $field (with value $value) is not a field," +
              s" ensure the field is specified as a function returning a field on an object of class ${this.getClass}," +
              s" eg '_.a' to select hypothetical bundle field 'a'"
          )
        )
        val valueBinding = value.topBindingOpt match {
          case Some(litBinding: LitBinding) => litBinding
          case _ => throw new BundleLiteralException(s"field $fieldName specified with non-literal value $value")
        }

        field match { // Get the litArg(s) for this field
          case field: Bits =>
            if (field.getClass != value.getClass) { // TODO typeEquivalent is too strict because it checks width
              throw new BundleLiteralException(
                s"Field $fieldName $field specified with non-type-equivalent value $value"
              )
            }
            val litArg = valueBinding match {
              case ElementLitBinding(litArg) => litArg
              case BundleLitBinding(litMap) =>
                litMap.getOrElse(
                  value,
                  throw new BundleLiteralException(s"Field $fieldName specified with unspecified value")
                )
              case VecLitBinding(litMap) =>
                litMap.getOrElse(
                  value,
                  throw new VecLiteralException(s"Vec literal $fieldName specified with out literal values")
                )

            }
            Seq(field -> litArg)

          case field: Record =>
            if (!(field.typeEquivalent(value))) {
              throw new BundleLiteralException(
                s"field $fieldName $field specified with non-type-equivalent value $value"
              )
            }
            // Copy the source BundleLitBinding with fields (keys) remapped to the clone
            val remap = getMatchedFields(value, field).toMap
            value.topBinding.asInstanceOf[BundleLitBinding].litMap.map {
              case (valueField, valueValue) =>
                remap(valueField) -> valueValue
            }

          case vecField: Vec[_] =>
            if (!(vecField.typeEquivalent(value))) {
              throw new BundleLiteralException(
                s"field $fieldName $vecField specified with non-type-equivalent value $value"
              )
            }
            // Copy the source BundleLitBinding with fields (keys) remapped to the clone
            val remap = getMatchedFields(value, vecField).toMap
            value.topBinding.asInstanceOf[VecLitBinding].litMap.map {
              case (valueField, valueValue) =>
                remap(valueField) -> valueValue
            }

          case field: EnumType => {
            if (!(field.typeEquivalent(value))) {
              throw new BundleLiteralException(
                s"field $fieldName $field specified with non-type-equivalent enum value $value"
              )
            }
            val litArg = valueBinding match {
              case ElementLitBinding(litArg) => litArg
              case _ =>
                throw new BundleLiteralException(s"field $fieldName $field could not be matched with $valueBinding")
            }
            Seq(field -> litArg)
          }
          case _ => throw new BundleLiteralException(s"unsupported field $fieldName of type $field")
        }
    }

    // don't convert to a Map yet to preserve duplicate keys
    val duplicates = bundleLitMap.map(_._1).groupBy(identity).collect { case (x, elts) if elts.size > 1 => x }
    if (!duplicates.isEmpty) {
      val duplicateNames = duplicates.map(cloneFields(_)).mkString(", ")
      throw new BundleLiteralException(s"duplicate fields $duplicateNames in Bundle literal constructor")
    }
    clone.bind(BundleLitBinding(bundleLitMap.toMap))
    clone
  }

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
    topBindingOpt match {
      case Some(BundleLitBinding(_)) =>
        val contents = elements.toList.reverse.map {
          case (name, data) =>
            s"$name=$data"
        }.mkString(", ")
        s"$className($contents)"
      case _ => stringAccessor(s"$className")
    }
  }

  def elements: SeqMap[String, Data]

  /** Name for Pretty Printing */
  def className: String = this.getClass.getSimpleName

  private[chisel3] override def typeEquivalent(that: Data): Boolean = that match {
    case that: Record =>
      this.getClass == that.getClass &&
        this.elements.size == that.elements.size &&
        this.elements.forall {
          case (name, model) =>
            that.elements.contains(name) &&
              (that.elements(name).typeEquivalent(model))
        }
    case _ => false
  }

  private[chisel3] override def _onModuleClose: Unit = {
    // This is usually done during binding, but these must still be set for unbound Records
    if (this.binding.isEmpty) {
      setElementRefs()
    }
  }

  private[chisel3] final def allElements: Seq[Element] = elements.toIndexedSeq.flatMap(_._2.allElements)

  override def getElements: Seq[Data] = elements.toIndexedSeq.map(_._2)

  // Helper because Bundle elements are reversed before printing
  private[chisel3] def toPrintableHelper(elts: Seq[(String, Data)]): Printable = {
    val xs =
      if (elts.isEmpty) List.empty[Printable] // special case because of dropRight below
      else
        elts.flatMap {
          case (name, data) =>
            List(PString(s"$name -> "), data.toPrintable, PString(", "))
        }.dropRight(1) // Remove trailing ", "
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

package experimental {

  class BundleLiteralException(message: String) extends ChiselException(message)
  class VecLiteralException(message: String) extends ChiselException(message)

}

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
  *      reg <> io.inPacket
  *      io.outPacket <> reg
  *   }
  * }}}
  */
abstract class Bundle(implicit compileOptions: CompileOptions) extends Record {
  assert(
    _usingPlugin,
    "The Chisel compiler plugin is now required for compiling Chisel code. " +
      "Please see https://github.com/chipsalliance/chisel3#build-your-own-chisel-projects."
  )

  override def className: String = this.getClass.getSimpleName match {
    case name if name.startsWith("$anon$") => "AnonymousBundle" // fallback for anonymous Bundle case
    case ""                                => "AnonymousBundle" // ditto, but on other platforms
    case name                              => name
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
  final lazy val elements: SeqMap[String, Data] = {
    val hardwareFields = _elementsImpl.flatMap {
      case (name, data: Data) =>
        if (data.isSynthesizable) {
          Some(s"$name: $data")
        } else {
          None
        }
      case (name, Some(data: Data)) =>
        if (data.isSynthesizable) {
          Some(s"$name: $data")
        } else {
          None
        }
      case (name, s: scala.collection.Seq[Any]) if s.nonEmpty =>
        s.head match {
          // Ignore empty Seq()
          case d: Data =>
            throwException(
              "Public Seq members cannot be used to define Bundle elements " +
                s"(found public Seq member '${name}'). " +
                "Either use a Vec if all elements are of the same type, or MixedVec if the elements " +
                "are of different types. If this Seq member is not intended to construct RTL, mix in the trait " +
                "IgnoreSeqInBundle."
            )
          case _ => // don't care about non-Data Seq
        }
        None

      case _ => None
    }
    if (hardwareFields.nonEmpty) {
      throw ExpectedChiselTypeException(s"Bundle: $this contains hardware fields: " + hardwareFields.mkString(","))
    }
    VectorMap(_elementsImpl.toSeq.flatMap {
      case (name, data: Data) =>
        Some(name -> data)
      case (name, Some(data: Data)) =>
        Some(name -> data)
      case _ => None
    }.sortWith {
      case ((an, a), (bn, b)) => (a._id > b._id) || ((a eq b) && (an > bn))
    }: _*)
  }
  /*
   * This method will be overwritten by the Chisel-Plugin
   */
  protected def _elementsImpl: SeqMap[String, Any] = {
    val nameMap = LinkedHashMap[String, Data]()
    for (m <- getPublicFields(classOf[Bundle])) {
      getBundleField(m) match {
        case Some(d: Data) =>
          // Checking for chiselType is done in elements method
          if (nameMap contains m.getName) {
            require(nameMap(m.getName) eq d)
          } else {
            nameMap(m.getName) = d
          }
        case None =>
          if (!ignoreSeq) {
            m.invoke(this) match {
              case s: scala.collection.Seq[Any] if s.nonEmpty =>
                s.head match {
                  // Ignore empty Seq()
                  case d: Data =>
                    throwException(
                      "Public Seq members cannot be used to define Bundle elements " +
                        s"(found public Seq member '${m.getName}'). " +
                        "Either use a Vec if all elements are of the same type, or MixedVec if the elements " +
                        "are of different types. If this Seq member is not intended to construct RTL, mix in the trait " +
                        "IgnoreSeqInBundle."
                    )
                  case _ => // don't care about non-Data Seq
                }
              case _ => // not a Seq
            }
          }
      }
    }
    VectorMap(nameMap.toSeq.sortWith { case ((an, a), (bn, b)) => (a._id > b._id) || ((a eq b) && (an > bn)) }: _*)
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

  /** Indicates if a concrete Bundle class was compiled using the compiler plugin
    *
    * Used for optimizing Chisel's performance and testing Chisel itself
    * @note This should not be used in user code!
    */
  protected def _usingPlugin: Boolean = false

  private def checkClone(clone: Bundle): Unit = {
    for ((name, field) <- elements) {
      if (clone.elements(name) eq field) {
        throw new AutoClonetypeException(
          s"Automatically cloned $clone has field '$name' aliased with base $this." +
            " In the future, this will be solved automatically by the compiler plugin." +
            " For now, ensure Chisel types used in the Bundle definition are passed through constructor arguments," +
            " or wrapped in Input(...), Output(...), or Flipped(...) if appropriate." +
            " As a last resort, you can override cloneType manually."
        )
      }
    }

  }

  override def cloneType: this.type = {
    val clone = _cloneTypeImpl.asInstanceOf[this.type]
    checkClone(clone)
    clone
  }

  /** Implementation of cloneType using runtime reflection. This should _never_ be overridden or called in user-code
    *
    * @note This is overridden by the compiler plugin (this implementation is never called)
    */
  protected def _cloneTypeImpl: Bundle = {
    throwException(
      s"Internal Error! This should have been implemented by the chisel3-plugin. Please file an issue against chisel3"
    )
  }

  /** Default "pretty-print" implementation
    * Analogous to printing a Map
    * Results in "`Bundle(elt0.name -> elt0.value, ...)`"
    * @note The order is reversed from the order of elements in order to print
    *   the fields in the order they were defined
    */
  override def toPrintable: Printable = toPrintableHelper(elements.toList.reverse)
}
