package chisel3.internal.operators

import chisel3._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform}

import scala.language.experimental.macros

private[chisel3] trait HasHead[T <: Data] extends Data {
    /** Head operator
    *
    * @param n the number of bits to take
    * @return The `n` most significant bits of this $coll
    * @group Bitwise
    */
  def head(n: Int): Data = throw new Exception("not impl")

  /** @group SourceInfoTransformMacro */
  private[chisel3] def do_head(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Data
}