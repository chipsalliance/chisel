package chisel3.internal.operators

import chisel3._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform}

import scala.language.experimental.macros

private[chisel3] trait HasStaticRightShift[T <: Data] extends Data {
  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  def >> (that: BigInt): Data = throw new Exception("not impl")

  /** @group SourceInfoTransformMacro */
  private[chisel3] def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Data

  /** Static right shift operator
    *
    * @param that an amount to shift by
    * @return this $coll with `that` many least significant bits truncated
    * $unchangedWidth
    * @group Bitwise
    */
  def >> (that: Int): Data = throw new Exception("not impl")

  /** @group SourceInfoTransformMacro */
  private[chisel3] def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Data
}