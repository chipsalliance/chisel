// See LICENSE for license details.
package chisel3

import chisel3.experimental.FixedPoint
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.{BinaryPoint, DefPrim, ILit, KnownBinaryPoint}
import chisel3.internal.firrtl.PrimOp._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal.throwException

// scalastyle:off method.name number.of.methods
trait NumBits[T <: Bits] {
  this: T =>

  /** @group SourceInfoTransformMacro */
  def do_=/= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, NotEqualOp, that)
  /** @group SourceInfoTransformMacro */
  def do_=== (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, EqualOp, that)

  /** @group SourceInfoTransformMacro */
  def do_asFixedPoint(binaryPoint: BinaryPoint)
                     (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
    binaryPoint match {
      case KnownBinaryPoint(value) =>
        val iLit = ILit(value)
        pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
      case _ =>
        throwException(
          s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint"
        )
    }
  }

  /** @group SourceInfoTransformMacro */
  def do_< (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, LessOp, that)
  /** @group SourceInfoTransformMacro */
  def do_> (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, GreaterOp, that)
  /** @group SourceInfoTransformMacro */
  def do_<= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, LessEqOp, that)
  /** @group SourceInfoTransformMacro */
  def do_>= (that: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool =
    compop(sourceInfo, GreaterEqOp, that)

}
