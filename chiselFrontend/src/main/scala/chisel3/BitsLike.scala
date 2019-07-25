// See LICENSE for license details.
package chisel3

import scala.language.experimental.macros
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.PrimOp.{EqualOp, GreaterEqOp, GreaterOp, LessEqOp, LessOp, NotEqualOp}
import chisel3.internal.firrtl.{DefPrim, ILit, PrimOp, Width}
import chisel3.internal.{Builder, requireIsHardware}
import chisel3.internal.sourceinfo.{SourceInfo, SourceInfoTransform, SourceInfoWhiteboxTransform}

// scalastyle:off method.name


// Bits derivatives, but not Bits itself implement BitsLike
private[chisel3] trait BitsLike[T<: Bits] {



//  private[chisel3] def unop(sourceInfo: SourceInfo, dest: T, op: PrimOp): T = {
//    requireIsHardware(this, "bits operated on")
//    pushOp(DefPrim(sourceInfo, dest, op, this.ref))
//  }
//  private[chisel3] def binop(sourceInfo: SourceInfo, dest: T, op: PrimOp, other: BigInt): T = {
//    requireIsHardware(this, "bits operated on")
//    pushOp(DefPrim(sourceInfo, dest, op, this.ref, ILit(other)))
//  }
//  private[chisel3] def binop(sourceInfo: SourceInfo, dest: T, op: PrimOp, other: T): T = {
//    requireIsHardware(this, "bits operated on")
//    requireIsHardware(other, "bits operated on")
//    pushOp(DefPrim(sourceInfo, dest, op, this.ref, other.ref))
//  }
//  private[chisel3] def binopUint(sourceInfo: SourceInfo, dest: T, op: PrimOp, other: UInt): T = {
//    requireIsHardware(this, "bits operated on")
//    requireIsHardware(other, "bits operated on")
//    pushOp(DefPrim(sourceInfo, dest, op, this.ref, other.ref))
//  }
//  private[chisel3] def redop(sourceInfo: SourceInfo, op: PrimOp): Bool = {
//    requireIsHardware(this, "bits operated on")
//    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref))
//  }
//  private[chisel3] def compop(sourceInfo: SourceInfo, op: PrimOp, other: T): Bool = {
//    requireIsHardware(this, "bits operated on")
//    requireIsHardware(other, "bits operated on")
//    pushOp(DefPrim(sourceInfo, Bool(), op, this.ref, other.ref))
//  }




}
