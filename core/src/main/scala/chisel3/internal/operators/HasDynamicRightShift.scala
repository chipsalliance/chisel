package chisel3.internal.operators

import chisel3._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform}

import scala.language.experimental.macros

private[chisel3] trait HasDynamicRightShift[T <: Data] extends Data {
  /** Dynamic right shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted right by the value of `that` component, inserting zeros into the most
    * significant bits.
    * $unchangedWidth
    * @group Bitwise
    */
  def >> (that: UInt): Data = throw new Exception("not impl")

  /** @group SourceInfoTransformMacro */
  private[chisel3] def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Data
}