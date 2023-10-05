// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.ir._

object CheckTypes {
  def legalResetType(tpe: Type): Boolean = tpe match {
    case UIntType(IntWidth(w)) if w == 1 => true
    case AsyncResetType                  => true
    case ResetType                       => true
    case UIntType(UnknownWidth)          =>
      // cannot catch here, though width may ultimately be wrong
      true
    case _ => false
  }

  // Check if it is legal for the source type to drive the sink type
  // Which is which matters because ResetType can be driven by itself, Bool, or AsyncResetType, but
  //   it cannot drive Bool nor AsyncResetType
  private def compare(sink: Type, source: Type): Boolean =
    (sink, source) match {
      case (_: UIntType, _: UIntType) => true
      case (_: SIntType, _: SIntType) => true
      case (ClockType, ClockType)           => true
      case (AsyncResetType, AsyncResetType) => true
      case (ResetType, tpe)                 => legalResetType(tpe)
      case (tpe, ResetType)                 => legalResetType(tpe)
      // Analog totally skips out of the Firrtl type system.
      // The only way Analog can play with another Analog component is through Attach.
      // Otherwise, we'd need to special case it during lowering.
      case (_: AnalogType, _: AnalogType) => false
      case (sink: VectorType, source: VectorType) =>
        sink.size == source.size && compare(sink.tpe, source.tpe)
      case (sink: BundleType, source: BundleType) =>
        (sink.fields.size == source.fields.size) &&
          sink.fields.zip(source.fields).forall {
            case (f1, f2) =>
              (f1.flip == f2.flip) && (f1.name == f2.name) && (f1.flip match {
                case Default => compare(f1.tpe, f2.tpe)
                // We allow UInt<1> and AsyncReset to drive Reset but not the other way around
                case Flip => compare(f2.tpe, f1.tpe)
              })
          }
      // Const connection validity is checked later on in the Firrtl compiler.
      case (sink: ConstType, source: ConstType) => compare(sink.underlying, source.underlying)
      case (sink, source: ConstType) => compare(sink, source.underlying)
      case (sink: ConstType, source) => compare(sink.underlying, source)
      case _ => false
    }

  def validConnect(locTpe: Type, expTpe: Type): Boolean = {
    compare(locTpe, expTpe)
  }
}
