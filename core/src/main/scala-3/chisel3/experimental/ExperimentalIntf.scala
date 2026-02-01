// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import chisel3._
import chisel3.internal.BaseModule

/** Clones an existing module and returns a record of all its top-level ports.
  * Each element of the record is named with a string matching the
  * corresponding port's name and shares the port's type.
  * @example {{{
  * val q1 = Module(new Queue(UInt(32.W), 2))
  * val q2_io = CloneModuleAsRecord(q1)("io").asInstanceOf[q1.io.type]
  * q2_io.enq <> q1.io.deq
  * }}}
  */
private[chisel3] trait CloneModuleAsRecord$Intf { self: CloneModuleAsRecord.type =>
  def apply(
    proto: BaseModule
  )(using sourceInfo: chisel3.experimental.SourceInfo): experimental.ClonePorts = _applyImpl(proto)
}

private[chisel3] trait AddBundleLiteralConstructorIntf[T <: Record] {
  self: BundleLiterals.AddBundleLiteralConstructor[T] =>

  def Lit(elems: (T => (Data, Data))*)(using sourceInfo: SourceInfo): T = _LitImpl(elems: _*)
}

private[chisel3] trait AddVecLiteralConstructorIntf[T <: Data] {
  self: VecLiterals.AddVecLiteralConstructor[T] =>

  /** Given a generator of a list tuples of the form [Int, Data]
    * constructs a Vec literal, parallel concept to `BundleLiteral`
    *
    * @param elems tuples of an index and a literal value
    * @return
    */
  def Lit(elems: (Int, T)*)(using sourceInfo: SourceInfo): Vec[T] = _LitImpl(elems: _*)
}

private[chisel3] trait AddObjectLiteralConstructorIntf {
  self: VecLiterals.AddObjectLiteralConstructor =>

  /** This provides an literal construction method for cases using
    * object `Vec` as in `Vec.Lit(1.U, 2.U)`
    */
  def Lit[T <: Data](elems: T*)(using sourceInfo: SourceInfo): Vec[T] = _LitImpl(elems: _*)
}
