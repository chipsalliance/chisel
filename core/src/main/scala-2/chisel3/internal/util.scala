// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import chisel3._
import scala.reflect.runtime.universe.{typeTag, TypeTag}

object util {
  // Workaround for https://github.com/chipsalliance/chisel/issues/4162
  // We can't use the .asTypeOf workaround because this is used to implement .asTypeOf
  private[chisel3] def _padHandleBool[A <: Bits](
    x:     A,
    width: Int
  )(
    implicit sourceInfo: SourceInfo,
    tag:                 TypeTag[A]
  ): A = x match {
    case b: Bool if !b.isLit && width > 1 && tag.tpe =:= typeTag[UInt].tpe =>
      val _pad = Wire(UInt(width.W))
      _pad := b
      _pad.asInstanceOf[A] // This cast is safe because we know A is UInt on this path
    case u => u.pad(width).asInstanceOf[A]
  }

  // Resize that to this width (if known)
  private[chisel3] def _resizeToWidth[A <: Bits: TypeTag](
    that:           A,
    targetWidthOpt: Option[Int]
  )(fromUInt:       UInt => A
  )(
    implicit sourceInfo: SourceInfo
  ): A = {
    (targetWidthOpt, that.widthOption) match {
      case (Some(targetWidth), Some(thatWidth)) =>
        if (targetWidth == thatWidth) that
        else if (targetWidth > thatWidth) _padHandleBool(that, targetWidth)
        else fromUInt(that.take(targetWidth))
      case (Some(targetWidth), None) => fromUInt(_padHandleBool(that, targetWidth).take(targetWidth))
      case (None, Some(thatWidth))   => that
      case (None, None)              => that
    }
  }
}
