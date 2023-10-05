// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.firrtl.{SLit, Width}

trait SIntFactory {

  /** Create an SInt type with inferred width. */
  def apply(): SInt = apply(Width())

  /** Create a SInt type or port with fixed width. */
  def apply(width: Width): SInt = new SInt(width)

  /** Create an SInt literal with specified width. */
  protected[chisel3] def Lit(value: BigInt, width: Width): SInt = {
    val lit = SLit(value, width)
    val result = new SInt(lit.width)
    lit.bindLitArg(result)
  }
}
