// See LICENSE for license details.

package chisel3

import chisel3.internal.firrtl.{IntervalRange, KnownWidth, ULit, UnknownWidth, Width}
import firrtl.Utils
import firrtl.constraint.IsKnown
import firrtl.ir.{Closed, IntWidth, Open}

// This is currently a factory because both Bits and UInt inherit it.
trait UIntFactory {
  /** Create a UInt type with inferred width. */
  def apply(): UInt = apply(Width())
  /** Create a UInt port with specified width. */
  def apply(width: Width): UInt = new UInt(width)

  /** Create a UInt literal with specified width. */
  // scalastyle:off method.name
  protected[chisel3] def Lit(value: BigInt, width: Width): UInt = {
    val lit = ULit(value, width)
    val result = new UInt(lit.width)
    // Bind result to being an Literal
    lit.bindLitArg(result)
  }
  /** Create a UInt with the specified range, validate that range is effectively > 0
    */
  //scalastyle:off cyclomatic.complexity
  def apply(range: IntervalRange): UInt = {
    // Check is only done against lower bound because range will already insist that range high >= low
    range.lowerBound match {
      case Closed(bound) if bound < 0 =>
        throw new ChiselException(s"Attempt to create UInt with closed lower bound of $bound, must be > 0")
      case Open(bound) if bound < -1 =>
        throw new ChiselException(s"Attempt to create UInt with open lower bound of $bound, must be > -1")
      case _ =>
    }

    // because this is a UInt we don't have to take into account the lower bound
    val newWidth = if(range.upperBound.isInstanceOf[IsKnown]) {
      KnownWidth(Utils.getUIntWidth(range.maxAdjusted.get).max(1))  // max(1) handles range"[0,0]"
    } else {
      UnknownWidth()
    }

    apply(newWidth)
  }
}
