// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.experimental.SourceInfo

private[chisel3] trait Property$Intf[T] { self: Property[T] =>

  /** Perform addition as defined by FIRRTL spec section Integer Add Operation.
    */
  final def +(
    that: Property[T]
  )(using ev: PropertyArithmeticOps[Property[T]], sourceInfo: SourceInfo): Property[T] =
    _addImpl(that)

  /** Perform multiplication as defined by FIRRTL spec section Integer Multiply Operation.
    */
  final def *(
    that: Property[T]
  )(using ev: PropertyArithmeticOps[Property[T]], sourceInfo: SourceInfo): Property[T] =
    _mulImpl(that)

  /** Perform shift right as defined by FIRRTL spec section Integer Shift Right Operation.
    */
  final def >>(
    that: Property[T]
  )(using ev: PropertyArithmeticOps[Property[T]], sourceInfo: SourceInfo): Property[T] =
    _shrImpl(that)

  /** Perform shift left as defined by FIRRTL spec section Integer Shift Left Operation.
    */
  final def <<(
    that: Property[T]
  )(using ev: PropertyArithmeticOps[Property[T]], sourceInfo: SourceInfo): Property[T] =
    _shlImpl(that)

  /** Perform concatenation as defined by FIRRTL spec section List Concatenation Operation.
    */
  final def ++(
    that: Property[T]
  )(using ev: PropertySequenceOps[Property[T]], sourceInfo: SourceInfo): Property[T] =
    _concatImpl(that)
}
