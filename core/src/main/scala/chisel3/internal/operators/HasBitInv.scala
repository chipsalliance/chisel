package chisel3.internal.operators

import chisel3._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform}

import scala.language.experimental.macros

private[chisel3] trait HasBitInv[T <: Data] extends Data {

  /** Bitwise inversion operator
    *
    * @return this $coll with each bit inverted
    * @group Bitwise
    */
  def unary_~ : Data = throw new Exception("not impl")

  @deprecated("Calling this function with an empty argument list is invalid in Scala 3. Use the form without parentheses instead", "Chisel 3.5")
  def unary_~(dummy: Int*): Data = throw new Exception("not impl")

  /** @group SourceInfoTransformMacro */
  private[chisel3] def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Data
}