package chisel3.internal.operators

import chisel3._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform}

import scala.language.experimental.macros

private[chisel3] trait HasDynamicLeftShift[T <: Data] extends Data {
  /** Dynamic left shift operator
    *
    * @param that a hardware component
    * @return this $coll dynamically shifted left by `that` many places, shifting in zeros from the right
    * @note The width of the returned $coll is `width of this + pow(2, width of that) - 1`.
    * @group Bitwise
    */
  def << (that: UInt): Data = throw new Exception("not impl")

  /** @group SourceInfoTransformMacro */
  private[chisel3] def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Data
}