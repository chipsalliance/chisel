// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros
import collection.mutable

import chisel3.internal._
import chisel3.internal.Builder.{pushCommand, pushOp}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{SourceInfo, DeprecatedSourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform,
  UIntTransform}
import chisel3.internal.firrtl.PrimOp._

private [core] trait UIntMacros { this: UInt =>
  // TODO: refactor to share documentation with Num or add independent scaladoc
  final def unary_- (): UInt = macro SourceInfoTransform.noArg
  final def unary_-% (): UInt = macro SourceInfoTransform.noArg
  final def * (that: SInt): SInt = macro SourceInfoTransform.thatArg
  final def +& (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def +% (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def -& (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def -% (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def & (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def | (that: UInt): UInt = macro SourceInfoTransform.thatArg
  final def ^ (that: UInt): UInt = macro SourceInfoTransform.thatArg
  // REVIEW TODO: Can this be defined on Bits?
  final def orR(): Bool = macro SourceInfoTransform.noArg
  final def andR(): Bool = macro SourceInfoTransform.noArg
  final def xorR(): Bool = macro SourceInfoTransform.noArg
  @chiselRuntimeDeprecated
  @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
  final def != (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this =/= that
  final def =/= (that: UInt): Bool = macro SourceInfoTransform.thatArg
  final def === (that: UInt): Bool = macro SourceInfoTransform.thatArg
  final def unary_! () : Bool = macro SourceInfoTransform.noArg
  final def bitSet(off: UInt, dat: Bool): UInt = macro UIntTransform.bitset
  final def zext(): SInt = macro SourceInfoTransform.noArg
}

private [core] trait UIntMethods { this: UInt =>
  private[core] override def typeEquivalent(that: Data): Boolean =
    that.isInstanceOf[UInt] && this.width == that.width
  private[core] override def cloneTypeWidth(w: Width): this.type =
    new UInt(w).asInstanceOf[this.type]
  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that.asUInt
  }
  def do_unary_- (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : UInt = 0.U - this
  def do_unary_-% (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = 0.U -% this
  override def do_+ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this +% that
  override def do_- (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this -% that
  override def do_/ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width), DivideOp, that)
  override def do_% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width), RemOp, that)
  override def do_* (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width + that.width), TimesOp, that)
  def do_* (that: SInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt = that * this
  def do_+& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt((this.width max that.width) + 1), AddOp, that)
  def do_+% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    (this +& that).tail(1)
  def do_-& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, SInt((this.width max that.width) + 1), SubOp, that).asUInt
  def do_-% (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    (this -& that).tail(1)
  def do_abs(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this
  def do_& (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitAndOp, that)
  def do_| (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitOrOp, that)
  def do_^ (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width max that.width), BitXorOp, that)
  def do_unary_~ (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    unop(sourceInfo, UInt(width = width), BitNotOp)
  def do_orR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this =/= 0.U
  def do_andR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = ~this === 0.U
  def do_xorR(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = redop(sourceInfo, XorReduceOp)
  override def do_< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessOp, that)
  override def do_> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterOp, that)
  override def do_<= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, LessEqOp, that)
  override def do_>= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, GreaterEqOp, that)
  override def do_<< (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width + that), ShiftLeftOp, validateShiftAmount(that))
  override def do_<< (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    this << that.toInt
  override def do_<< (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width.dynamicShiftLeft(that.width)), DynamicShiftLeftOp, that)
  override def do_>> (that: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width.shiftRight(that)), ShiftRightOp, validateShiftAmount(that))
  override def do_>> (that: BigInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    this >> that.toInt
  override def do_>> (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt =
    binop(sourceInfo, UInt(this.width), DynamicShiftRightOp, that)
  def do_=/= (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, NotEqualOp, that)
  def do_=== (that: UInt)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = compop(sourceInfo, EqualOp, that)
  def do_unary_! (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions) : Bool = this === 0.U(1.W)
  def do_bitSet(off: UInt, dat: Bool)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    val bit = 1.U(1.W) << off
    Mux(dat, this | bit, ~(~this | bit))
  }
  def do_zext(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width + 1), ConvertOp, ref))
  override def do_asSInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SInt =
    pushOp(DefPrim(sourceInfo, SInt(width), AsSIntOp, ref))
  override def do_asUInt(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = this
  override def do_asFixedPoint(binaryPoint: BinaryPoint)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): FixedPoint = {
    binaryPoint match {
      case KnownBinaryPoint(value) =>
        val iLit = ILit(value)
        pushOp(DefPrim(sourceInfo, FixedPoint(width, binaryPoint), AsFixedPointOp, ref, iLit))
      case _ =>
        throwException(s"cannot call $this.asFixedPoint(binaryPoint=$binaryPoint), you must specify a known binaryPoint")
    }
  }
}
