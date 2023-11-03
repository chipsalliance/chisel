// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.reflect.DataMirror
import chisel3.util.{log2Up, BitPat, Mux1H, PriorityMux}
import chisel3.util.experimental.decode.{decoder, TruthTable}
import chisel3.util.experimental.InlineInstance
import scala.collection.immutable.VectorMap
import scala.collection.mutable.HashSet

import SparseVec.{DefaultValueBehavior, Lookup, OutOfBoundsBehavior}

/** A sparse vector.  Under the hood, this is a [[Record]] that can be
  * dynamically indexed as if it were a dense [[Vec]].
  *
  * [[SparseVec]] has the usual trappings of a [[Vec]].  It has a `size` and a
  * `gen` type.  However, it also has an `indices` argument.  This indicates the
  * indices at which the [[SparseVec]] is allowed to have data.  Additionally,
  * the behavior of a [[SparseVec]] around what happens if a value is read from
  * a value not in the `indices`:
  *
  * - `defaultValue` sets the default value that is read from an index between
  *   the zeroth index and the largest value in `indices` that is not in the
  *   `indices`.
  * - `outOfBoundsValue` sets the behavior when reading a value larger than the
  *   largest value in `indices`.
  *
  * The reason for this configurability is to enable exact compatibility with an
  * equivalent dense [[Vec]] of the same size and initialized to a given value.
  * Specifically, use [[SparseVec.DefaultValueBehavior.DynamicIndexEquivalent]]
  * and [[SparseVec.OutOfBoundsBehavior.First]] to make this behave as such:
  *
  * 1. The [[SparseVec]] has a default value of how a FIRRTL compiler compiles
  *    [[DontCare]] for a dynamic index.
  *
  * 2. The [[SparseVec]] out-of-bounds behavior returns the zeroth element if a
  *    zeroth element exists.  Otherwise, it returns a [[DontCare]].
  *
  * Note that this [[DontCare]] is likely not a true "don't care" that will be
  * optimized to any value.  Instead, it is a value equal to how a FIRRTL
  * compiler chooses to optimize a dynamic index into a wire vector initialized
  * with a [[DontCare]].  This has historically been zero.
  *
  * Once created, a [[SparseVec]] can be written or read from as a [[Record]].
  * It may also be read from using a dynamic index, but not written to.  Neither
  * the default value nor the out-of-bounds value may be written to.  The
  * dynamic index type is conifgurable and may be one of:
  *
  * - [[SparseVec.Lookup.Binary]] to convert the [[SparseVec]] index into a
  *   binary index into a dense vector.
  * - [[SparseVec.Lookup.OneHot]] to convert the [[SparseVec]] index into a
  *   one-hot encoded index into a dense vector using [[Mux1H]].
  * - [[SparseVec.Lookup.IfElse]] to use a sequence of [[when]] statements.
  *
  * A [[SparseVec]] will take up storage equal to the size of the provided
  * mapping argument with one additional slot for the default value, if one is
  * needed.
  *
  * @param size the apparent size of the vector
  * @param gen the element type of the vector
  * @param indices the indices of the vector which are valid
  * @param defaultValue the default value behavior when accessing an index not
  * in the `indices`
  * @param outOfBoundsValue the out-of-bounds behavior when accessing an index
  * larger than the largest value in `indices`
  */
class SparseVec[A <: Data](
  size:             Int,
  gen:              => A,
  indices:          Seq[Int],
  defaultValue:     DefaultValueBehavior.Type = DefaultValueBehavior.Indeterminate,
  outOfBoundsValue: OutOfBoundsBehavior.Type = OutOfBoundsBehavior.Indeterminate)
    extends Record {

  require(indices.size <= size, s"the SparseVec indices size (${indices.size}) must be <= the SparseVec size ($size)")

  // Populate the elements while simultaneously checking if the provided indices
  // is not unique Additionally, check and error if the same index is specified
  // twice.
  override final val elements = {
    var nonUniqueIndices: List[Int] = Nil // List is cheap in common case, no allocation
    val duplicates:       HashSet[Int] = HashSet.empty[Int]
    val result = indices.view.map {
      case index =>
        if (!duplicates.add(index))
          nonUniqueIndices ::= index
        index.toString -> DataMirror.internal.chiselTypeClone(gen)
    }.to(VectorMap)
    // Throw a runtime exception if there is a non-unique indices.
    // TODO: Improve this error message.
    if (nonUniqueIndices.nonEmpty) {
      throw new ChiselException(
        "Non-unique indices in SparseVec, got duplicates " + nonUniqueIndices.reverse.mkString(",")
      )
    }
    result
  }

  // A zeroValue is the value of the vector at index zero.  The zeroValue is
  // important because it will set the out-of-bounds behavior of the SparseVec.
  private val zeroValue: Option[Data] = elements.get("0")

  // Determine if a default value needs to exist.  This should only exist if
  // there are fewer map items than the size of the Vec.
  private val hasDefaultValue: Boolean = indices.size != size

  /** An alternative constructure to [[SparseVec]] where the size of the vector is
    * automatically set to the maximum value in the indices.
    *
    * @param gen the element type of the vector
    * @param indices the indices of the vector which are valid
    * @param defaultValue the default value behavior when accessing an index not
    * in the `indices`
    * @param outOfBoundsValue the out-of-bounds behavior when accessing an index
    * larger than the largest value in `indices`
    */
  final def this(
    gen:              => A,
    indices:          Seq[Int],
    defaultValue:     DefaultValueBehavior.Type,
    outOfBoundsValue: OutOfBoundsBehavior.Type
  ) = this(indices.max, gen, indices, defaultValue, outOfBoundsValue)

  /** Read a value from a [[SparseVec]] using one of several possible lookup
    * types. The returned value is read-only.
    *
    * @param addr the address of the value to read from the vec
    * @param lookupType the type of lookup, e.g., binary, one-hot, or when-based
    * @param sourceinfo implicit source locator information
    * @return a read-only value from the specified address
    * @throws ChiselException if the returned value is written to
    */
  def apply(addr: UInt, lookupType: Lookup.Type = Lookup.Binary)(implicit sourceinfo: SourceInfo): A = {
    val result: A = lookupType match {
      // Short circuit path if the indices is empty.  Return the default value.
      // A default value must exist.
      case d if indices.size == 0 =>
        WireInit(gen, defaultValue.getValue(d)).asInstanceOf[A]

      // Generate a lookup using a decoder.  Do this by creating a dense Vec ordered as so:
      //
      //   - Optional default value (if one exists)
      //   - 1st index element
      //   - 2nd index element
      //   - 3rd index element
      //   - ...
      //   - Nth index element
      //
      // Then create a decoder that converts an index into the sparse vector
      // (the full-sized index) to an index into the dense vector.  The way in
      // which the indexing into the dense vector is done is controlled by
      // Lookup.Decoder methods.  If the out-of-bounds behavior indicates that
      // the first element should be returned, then the decoder has additional
      // constraints such that an out-of-bounds value will point to the first
      // element in the denseVec.
      //
      // Practically speaking, for the LookupType.Binary, this is doing:
      //
      //   Index -> DenseVecIndex -> DenseVec(index)
      //
      // For the LookupType.OneHot, this is doing:
      //
      //   Index -> DenseVecOneHotIndex -> Mux1H(DenseVecOneHotIndex, DenseVec)
      //
      case d: Lookup.Decoder =>
        // The number of bits required to represent all addresses of the sparse
        // vector.
        val addrWidth = log2Up(size)

        // The address where user-provided data starts.  If there is no default
        // value needed, then start user data at address zero.  Otherwise, user
        // data starts at address one.
        val baseEncodedAddress = hasDefaultValue match {
          case true  => 1
          case false => 0
        }

        // The total number of output addresses that need to be indexed.
        val encodedSize = baseEncodedAddress + elements.size

        // The dense vector of optional default value and all user-specified
        // indices values.
        val denseVec = VecInit(
          (Option.when(hasDefaultValue)(
            WireInit(gen, defaultValue.getValue(d)).asInstanceOf[A]
          ) ++ elements.values).toSeq
        )

        // Build up a sequence of BitPats that map the addresses to the encoding
        // in the denseVec.  If out-of-bounds values are possible and the user
        // instructed us to return the first-element on out-of-bounds acesses,
        // then fill out the array with BitPats to do this.
        val bitPats = indices.zipWithIndex.map {
          case (index, i) =>
            BitPat(index.U(addrWidth.W)) -> BitPat(d.encoding(i + baseEncodedAddress, encodedSize))
        } ++ {
          (outOfBoundsValue, zeroValue) match {
            case (OutOfBoundsBehavior.Indeterminate, _) | (_, None) => Seq.empty
            case (OutOfBoundsBehavior.First, Some(_)) =>
              (size until BigInt(addrWidth).pow(2).toInt).map {
                case i =>
                  BitPat(i.U(addrWidth.W)) -> BitPat(d.encoding(baseEncodedAddress, encodedSize))
              }
          }
        }

        // Generate the truth table.  If this SparseVec does not have a default
        // value or if the DefaultValue is indeterminate, then treat the default
        // value as don't care---any value indexed into the dense vector is
        // fine.  Otherwise, the default value exists and is always encoded as
        // address zero.
        val ttable = TruthTable(
          bitPats,
          (hasDefaultValue, defaultValue) match {
            case (false, _) | (_, DefaultValueBehavior.Indeterminate) => BitPat.dontCare(encodedSize)
            case _                                                    => BitPat(d.encoding(0, encodedSize))
          }
        )

        // Use a decoder to generate a lookup into the denseVec using the
        // provided Lookup.Decoder strategy.  Return the value.
        d.lookup(
          decoder(addr, ttable),
          denseVec
        ).asInstanceOf[A]

      // Generate a lookup into the elements of the SparseVec using when
      // statements.
      case Lookup.IfElse =>
        // The result of the lookup.
        val result = Wire(gen)

        // The default value _must_ be specified to bypass initialization checking.
        result := defaultValue.getValue(lookupType)

        // Generate one when statement for each value.
        indices.zip(elements.values).foreach {
          case (index, data) =>
            when(addr === index.U) {
              result := data
            }
        }

        // If the elements have a value at index zero and if the user indicated
        // they want to return this value on out-of-bounds behavior, then return
        // it.  Otherwise, add no logic for this case.
        (outOfBoundsValue, zeroValue) match {
          case (SparseVec.OutOfBoundsBehavior.Indeterminate, _) | (_, None) =>
          case (SparseVec.OutOfBoundsBehavior.First, Some(data)) =>
            when(addr >= size.U) {
              result := data
            }
        }

        // Return the result wire.
        result
    }

    // Return a read-only value to avoid users trying to write to a returned
    // wire and it getting write-holed.
    Detail.readOnly[A](result)
  }

  // This object contains implementation details for this class.
  private object Detail {

    // A passthrough module used to realize the readonly method.
    class ReadOnlyModule[A <: Data](gen: A) extends RawModule with InlineInstance {
      val in = IO(Flipped(gen))
      val out = IO(gen)
      out :<>= in
    }

    // TODO: This method produces sub-par error messages. Replace usages of this
    // with first-class Chisel support once it lands.
    def readOnly[A <: Data](
      gen: A
    ): A = {
      val readOnlyModule = Module(new ReadOnlyModule(chiselTypeOf(gen)))
      readOnlyModule.in :<>= gen
      readOnlyModule.out
    }

  }

}

/** Utilities related to [[SparseVec]]. */
object SparseVec {

  object Lookup {

    /** The root type of how a [[SparseVec]] can be accessed. */
    sealed trait Type

    /** A [[SparseVec$]] that is accessed using a [[chisel3.util.experimental.decode.decoder]]. */
    sealed trait Decoder extends Type {
      def encoding(index: Int, width: Int): UInt

      def lookup[A <: Data](
        index:  UInt,
        values: VecLike[A]
      )(
        implicit sourceinfo: SourceInfo
      ): A
    }

    /** A [[SparseVec$]] accessor that uses a binary-encoded lookup. */
    case object Binary extends Decoder {
      override final def encoding(index: Int, width: Int) = index.U(log2Up(width).W)

      override final def lookup[A <: Data](
        index:  UInt,
        values: VecLike[A]
      )(
        implicit sourceinfo: SourceInfo
      ) = values(index)
    }

    /** A [[SparseVec$]] accessor that uses a one-hot-encoded lookup. */
    case object OneHot extends Decoder {
      override final def encoding(index: Int, width: Int) = (BigInt(1) << index).U((width).W)

      override final def lookup[A <: Data](
        index:  UInt,
        values: VecLike[A]
      )(
        implicit sourceinfo: SourceInfo
      ) = Mux1H(index, values)
    }

    /** A [[SparseVec$]] accessor that uses Chisel's [[when]] abstraction for lookup. */
    case object IfElse extends Type

  }

  object DefaultValueBehavior {

    /** A type that specifies what the default value of a [[SparseVec]] is. */
    sealed trait Type {
      def getValue(lookupType: Lookup.Type): Data
    }

    /** Return the same result as if this were a dynamic index initialized to [[DontCare]]. */
    case object DynamicIndexEquivalent extends Type {
      override final def getValue(lookupType: Lookup.Type) = lookupType match {
        case _: Lookup.Decoder => DontCare
        case Lookup.IfElse => 0.U
      }
    }

    /** Allow Chisel and FIRRTL compilers to take advantage of undefined behavior. Any value may be returned. */
    case object Indeterminate extends Type {
      override final def getValue(lookupType: Lookup.Type) = lookupType match {
        case _: Lookup.Decoder => DontCare
        case Lookup.IfElse => DontCare
      }
    }

    /** Return a user-specified value. */
    case class UserSpecified(value: UInt) extends Type {
      override final def getValue(lookupType: Lookup.Type) = value
    }

  }

  object OutOfBoundsBehavior {

    /** A type that specifies what the out-of-bounds behavior of a [[SparseVec]] is. */
    sealed trait Type

    /** Return the first element of the [[SparseVec]] if one exists. */
    case object First extends Type

    /** Allow Chisel and FIRRTL compilers to take advantage of undefined behavior.  Any value may be returned. */
    case object Indeterminate extends Type

  }

}
