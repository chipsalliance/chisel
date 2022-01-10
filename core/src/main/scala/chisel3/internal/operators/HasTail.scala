package chisel3.internal.operators

import chisel3._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform}

import scala.language.experimental.macros

private[chisel3] trait HasTail[T <: Data] extends Data {
  /** Tail operator
    *
    * @param n the number of bits to remove
    * @return This $coll with the `n` most significant bits removed.
    * @group Bitwise
    */
  def tail(n: Int): Data = throw new Exception("not impl")

  /** @group SourceInfoTransformMacro */
  private[chisel3] def do_tail(n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Data
}