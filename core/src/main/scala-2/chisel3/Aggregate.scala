// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros
import chisel3.experimental.SourceInfo
import chisel3.internal.HasId
import chisel3.internal.sourceinfo.{SourceInfoTransform, VecTransform}

/** An abstract class for data types that solely consist of (are an aggregate
  * of) other Data objects.
  */
sealed trait Aggregate extends AggregateImpl

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
sealed class Vec[T <: Data] private[chisel3] (gen: => T, length: Int)
    extends VecImpl[T](gen, length)
    with VecLike[T]
    with Aggregate {

  override def toString: String = super[VecImpl].toString

  override def do_apply(p: UInt)(implicit sourceInfo: SourceInfo): T = _applyImpl(p)

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
    implicit sourceInfo: SourceInfo
  ): T = _reduceTreeImpl(redOp, layerOp)
}

object VecInit extends VecInitImpl with SourceInfoDoc {

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
  def do_apply[T <: Data](elts: Seq[T])(implicit sourceInfo: SourceInfo): Vec[T] = _applyImpl(elts)

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
  def do_apply[T <: Data](elt0: T, elts: T*)(implicit sourceInfo: SourceInfo): Vec[T] = _applyImpl(elt0, elts: _*)

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
    implicit sourceInfo: SourceInfo
  ): Vec[T] = _tabulateImpl(n)(gen)

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
    implicit sourceInfo: SourceInfo
  ): Vec[Vec[T]] = _tabulateImpl(n, m)(gen)

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
    implicit sourceInfo: SourceInfo
  ): Vec[Vec[Vec[T]]] = _tabulateImpl(n, m, p)(gen)

  /** Creates a new [[Vec]] of length `n` composed of the result of the given
    * function applied to an element of data type T.
    *
    * @param n number of elements in the vector
    * @param gen function that takes in an element T and returns an output
    * element of the same type
    */
  def fill[T <: Data](n: Int)(gen: => T): Vec[T] = macro VecTransform.fill

  /** @group SourceInfoTransformMacro */
  def do_fill[T <: Data](n: Int)(gen: => T)(implicit sourceInfo: SourceInfo): Vec[T] = _fillImpl(n)(gen)

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
    implicit sourceInfo: SourceInfo
  ): Vec[Vec[T]] = _fillImpl(n, m)(gen)

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
    implicit sourceInfo: SourceInfo
  ): Vec[Vec[Vec[T]]] = _fillImpl(n, m, p)(gen)

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
    implicit sourceInfo: SourceInfo
  ): Vec[T] = _iterateImpl(start, len)(f)
}

/** A trait for [[Vec]]s containing common hardware generators for collection
  * operations.
  */
trait VecLike[T <: Data] extends VecLikeImpl[T] with SourceInfoDoc {

  /** Creates a dynamically indexed read or write accessor into the array.
    */
  def apply(p: UInt): T = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_apply(p: UInt)(implicit sourceInfo: SourceInfo): T

  /** Outputs true if p outputs true for every element.
    */
  def forall(p: T => Bool): Bool = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_forall(p: T => Bool)(implicit sourceInfo: SourceInfo): Bool = _forallImpl(p)

  /** Outputs true if p outputs true for at least one element.
    */
  def exists(p: T => Bool): Bool = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_exists(p: T => Bool)(implicit sourceInfo: SourceInfo): Bool = _existsImpl(p)

  /** Outputs true if the vector contains at least one element equal to x (using
    * the === operator).
    */
  def contains(x: T)(implicit ev: T <:< UInt): Bool = macro VecTransform.contains

  /** @group SourceInfoTransformMacro */
  def do_contains(x: T)(implicit sourceInfo: SourceInfo, ev: T <:< UInt): Bool = _containsImpl(x)

  /** Outputs the number of elements for which p is true.
    */
  def count(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_count(p: T => Bool)(implicit sourceInfo: SourceInfo): UInt = _countImpl(p)

  /** Outputs the index of the first element for which p outputs true.
    */
  def indexWhere(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_indexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo): UInt = _indexWhereImpl(p)

  /** Outputs the index of the last element for which p outputs true.
    */
  def lastIndexWhere(p: T => Bool): UInt = macro SourceInfoTransform.pArg

  /** @group SourceInfoTransformMacro */
  def do_lastIndexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo): UInt = _lastIndexWhereImpl(p)

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
  def do_onlyIndexWhere(p: T => Bool)(implicit sourceInfo: SourceInfo): UInt = _onlyIndexWhereImpl(p)
}

/** Base class for Aggregates based on key values pairs of String and Data
  *
  * Record should only be extended by libraries and fairly sophisticated generators.
  * RTL writers should use [[Bundle]].  See [[Record#elements]] for an example.
  */
abstract class Record extends RecordImpl with Aggregate
