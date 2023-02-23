// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.firrtl.{KnownWidth, ULit, UnknownWidth, Width}
import firrtl.Utils

// This is currently a factory because both Bits and UInt inherit it.
trait UIntFactory {

  /** Create a UInt type with inferred width. */
  def apply(): UInt = apply(Width())

  /** Create a UInt port with specified width. */
  def apply(width: Width): UInt = new UInt(width)

  /** Create a UInt literal with specified width. */
  protected[chisel3] def Lit(value: BigInt, width: Width): UInt = {
    val lit = ULit(value, width)
    val result = new UInt(lit.width)
    // Bind result to being an Literal
    lit.bindLitArg(result)
  }

}
