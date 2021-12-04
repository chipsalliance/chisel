package chisel3.internal.operators

import chisel3._
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform}

import scala.language.experimental.macros

private[chisel3] trait HasPad[T <: Data] extends Data {
  /** Pad operator
    *
    * @param that the width to pad to
    * @return this @coll zero padded up to width `that`. If `that` is less than the width of the original component,
    * this method returns the original component.
    * @note For [[SInt]]s only, this will do sign extension.
    * @group Bitwise
    */
  def pad(that: Int): Data = throw new Exception("not impl")

  /** @group SourceInfoTransformMacro */
  private[chisel3] def do_pad(that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Data
}