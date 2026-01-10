// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.VecLiterals.AddVecLiteralConstructor

import scala.collection.immutable.{SeqMap, VectorMap}
import scala.collection.mutable.{HashSet, LinkedHashMap}
import chisel3.experimental.{BaseModule, BundleLiteralException, OpaqueType, SourceInfo, VecLiteralException}
import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._

import java.lang.Math.{floor, log10, pow}
import scala.collection.mutable

private[chisel3] trait VecIntf[T <: Data] { self: Vec[T] =>

  def apply(using SourceInfo)(p: UInt): T = _applyImpl(p)

  /** A reduce operation in a tree like structure instead of sequentially
    * @example An adder tree
    * {{{
    * val sumOut = inputNums.reduceTree((a: T, b: T) => (a + b))
    * }}}
    */
  // def reduceTree(redOp: (T, T) => T): T = macro VecTransform.reduceTreeDefault

  /** A reduce operation in a tree like structure instead of sequentially
    * @example A pipelined adder tree
    * {{{
    * val sumOut = inputNums.reduceTree(
    *   (a: T, b: T) => RegNext(a + b),
    *   (a: T) => RegNext(a)
    * )
    * }}}
    */
  def reduceTree(using SourceInfo)(redOp: (T, T) => T, layerOp: T => T = (x: T) => x): T = _reduceTreeImpl(redOp, layerOp)
}

private[chisel3] trait VecInit$Intf extends SourceInfoDoc { self: VecInit.type =>

  /** Creates a new [[Vec]] composed of elements of the input Seq of [[Data]]
    * nodes.
    *
    * @note input elements should be of the same type (this is checked at the
    * FIRRTL level, but not at the Scala / Chisel level)
    * @note the width of all output elements is the width of the largest input
    * element
    * @note output elements are connected from the input elements
    */
  def apply[T <: Data](using SourceInfo)(elts: Seq[T]): Vec[T] = _applyImpl(elts)

  /** Creates a new [[Vec]] composed of the input [[Data]] nodes.
    *
    * @note input elements should be of the same type (this is checked at the
    * FIRRTL level, but not at the Scala / Chisel level)
    * @note the width of all output elements is the width of the largest input
    * element
    * @note output elements are connected from the input elements
    */
  def apply[T <: Data](using SourceInfo)(elt0: T, elts: T*): Vec[T] = _applyImpl(elt0, elts: _*)

  /** Creates a new [[Vec]] of length `n` composed of the results of the given
    * function applied over a range of integer values starting from 0.
    *
    * @param n number of elements in the vector (the function is applied from
    * 0 to `n-1`)
    * @param gen function that takes in an Int (the index) and returns a
    * [[Data]] that becomes the output element
    */
  def tabulate[T <: Data](using SourceInfo)(n: Int)(gen: Int => T): Vec[T] = _tabulateImpl(n)(gen)

  /** Creates a new 2D [[Vec]] of length `n by m` composed of the results of the given
    * function applied over a range of integer values starting from 0.
    *
    * @param n number of 1D vectors inside outer vector
    * @param m number of elements in each 1D vector (the function is applied from
    * 0 to `n-1`)
    * @param gen function that takes in an Int (the index) and returns a
    * [[Data]] that becomes the output element
    */
  def tabulate[T <: Data](using SourceInfo)(n: Int, m: Int)(gen: (Int, Int) => T): Vec[Vec[T]] = _tabulateImpl(n, m)(gen)

  /** Creates a new 3D [[Vec]] of length `n by m by p` composed of the results of the given
    * function applied over a range of integer values starting from 0.
    *
    * @param n number of 2D vectors inside outer vector
    * @param m number of 1D vectors in each 2D vector
    * @param p number of elements in each 1D vector
    * @param gen function that takes in an Int (the index) and returns a
    * [[Data]] that becomes the output element
    */
  def tabulate[T <: Data](using SourceInfo)(n: Int, m: Int, p: Int)(gen: (Int, Int, Int) => T): Vec[Vec[Vec[T]]] = _tabulateImpl(n, m, p)(gen)

  /** Creates a new [[Vec]] of length `n` composed of the result of the given
    * function applied to an element of data type T.
    *
    * @param n number of elements in the vector
    * @param gen function that takes in an element T and returns an output
    * element of the same type
    */
  def fill[T <: Data](using SourceInfo)(n: Int)(gen: => T): Vec[T] = _fillImpl(n)(gen)

  /** Creates a new 2D [[Vec]] of length `n by m` composed of the result of the given
    * function applied to an element of data type T.
    *
    * @param n number of inner vectors (rows) in the outer vector
    * @param m number of elements in each inner vector (column)
    * @param gen function that takes in an element T and returns an output
    * element of the same type
    */
  def fill[T <: Data](using SourceInfo)(n: Int, m: Int)(gen: => T): Vec[Vec[T]] = _fillImpl(n, m)(gen)

  /** Creates a new 3D [[Vec]] of length `n by m by p` composed of the result of the given
    * function applied to an element of data type T.
    *
    * @param n number of 2D vectors inside outer vector
    * @param m number of 1D vectors in each 2D vector
    * @param p number of elements in each 1D vector
    * @param gen function that takes in an element T and returns an output
    * element of the same type
    */
  def fill[T <: Data](using SourceInfo)(n: Int, m: Int, p: Int)(gen: => T): Vec[Vec[Vec[T]]] = _fillImpl(n, m, p)(gen)

  /** Creates a new [[Vec]] of length `n` composed of the result of the given
    * function applied to an element of data type T.
    *
    * @param start First element in the Vec
    * @param len Lenth of elements in the Vec
    * @param f Function that applies the element T from previous index and returns the output
    * element to the next index
    */
  def iterate[T <: Data](using SourceInfo)(start: T, len: Int)(f: T => T): Vec[T] = _iterateImpl(start, len)(f)
}

private[chisel3] trait VecLikeImpl[T <: Data] extends SourceInfoDoc { self: VecLike[T] =>

  /** Creates a dynamically indexed read or write accessor into the array.
    */
  def apply(p: UInt)(using SourceInfo): T

  /** Outputs true if p outputs true for every element.
    */
  def forall(using SourceInfo)(p: T => Bool): Bool = _forallImpl(p)

  /** Outputs true if p outputs true for at least one element.
    */
  def exists(using SourceInfo)(p: T => Bool): Bool = _existsImpl(p)

  /** Outputs true if the vector contains at least one element equal to x (using
    * the === operator).
    */
  def contains(using SourceInfo)(x: T)(using T <:< UInt): Bool = _containsImpl(x)

  /** Outputs the number of elements for which p is true.
    */
  def count(using SourceInfo)(p: T => Bool): UInt = _countImpl(p)

  /** Outputs the index of the first element for which p outputs true.
    */
  def indexWhere(using SourceInfo)(p: T => Bool): UInt = _indexWhereImpl(p)

  /** Outputs the index of the last element for which p outputs true.
    */
  def lastIndexWhere(using SourceInfo)(p: T => Bool): UInt = _lastIndexWhereImpl(p)

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
  def onlyIndexWhere(using SourceInfo)(p: T => Bool): UInt = _onlyIndexWhereImpl(p)
}
