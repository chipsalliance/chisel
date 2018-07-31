// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.core.{Data, requireIsChiselType, requireIsHardware}
import chisel3.internal.naming.chiselName

import scala.collection.immutable.ListMap

object MixedVecInit {
  /**
    * Construct a new wire with the given bound values.
    * This is analogous to [[chisel3.core.VecInit]].
    * @param vals Values to create a MixedVec with and assign
    * @return MixedVec with given values assigned
    */
  def apply[T <: Data](vals: Seq[T]): MixedVec[T] = {
    // Create a wire of this type.
    val hetVecWire = Wire(MixedVec(vals.map(_.cloneTypeFull)))
    // Assign the given vals to this new wire.
    for ((a, b) <- hetVecWire.zip(vals)) {
      a := b
    }
    hetVecWire
  }
}

object MixedVec {
  /**
    * Create a MixedVec from that holds the given types.
    * @param eltsIn Element types. Must be Chisel types.
    * @return MixedVec with the given types.
    */
  def apply[T <: Data](eltsIn: Seq[T]): MixedVec[T] = new MixedVec(eltsIn)

  /**
    * Create a MixedVec from the type of the given Vec.
    * For example, given a Vec(2, UInt(8.W)), this creates MixedVec(Seq.fill(2){UInt(8.W)}).
    * @param vec Vec to use as template
    * @return MixedVec analogous to the given Vec.
    */
  def apply[T <: Data](vec: Vec[T]): MixedVec[T] = {
    MixedVec(Seq.fill(vec.length)(vec.sample_element))
  }
}

/**
  * A hardware array of elements that can hold values of different types/widths,
  * unlike Vec which can only hold elements of the same type/width.
  *
  * @param eltsIn Element types. Must be Chisel types.
  *
  * @example {{{
  * val v = Wire(MixedVec(Seq(UInt(8.W), UInt(16.W), UInt(32.W))))
  * v(0) := 100.U(8.W)
  * v(1) := 10000.U(16.W)
  * v(2) := 101.U(32.W)
  * }}}
  */
@chiselName
final class MixedVec[T <: Data](private val eltsIn: Seq[T]) extends Record with collection.IndexedSeq[T] {
  // We want to create MixedVec only with Chisel types.
  if (compileOptions.declaredTypeMustBeUnbound) {
    eltsIn.foreach(e => requireIsChiselType(e))
  }

  // Clone the inputs so that we have our own references.
  private val elts: IndexedSeq[T] = eltsIn.map(_.cloneTypeFull).toIndexedSeq

  /**
    * Statically (elaboration-time) retrieve the element at the given index.
    * @param index Index with which to retrieve.
    * @return Retrieved index.
    */
  def apply(index: Int): T = elts(index)

  /** Strong bulk connect, assigning elements in this MixedVec from elements in a Seq.
    *
    * @note the lengths of this and that must match
    */
  def :=(that: Seq[T]): Unit = {
    require(this.length == that.length)
    for ((a, b) <- this zip that)
      a := b
  }

  /**
    * Get the length of this MixedVec.
    * @return Number of elements in this MixedVec.
    */
  def length: Int = elts.length

  override val elements = ListMap(elts.zipWithIndex.map { case (element, index) => (index.toString, element) }: _*)

  // Need to re-clone again since we could have been bound since object creation.
  override def cloneType: this.type = MixedVec(elts.map(_.cloneTypeFull)).asInstanceOf[this.type]

  // IndexedSeq has its own hashCode/equals that we must not use
  override def hashCode: Int = super[Record].hashCode

  override def equals(that: Any): Boolean = super[Record].equals(that)
}
