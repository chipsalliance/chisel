// See LICENSE for license details.

package chisel3

import chisel3.internal.firrtl.{ULit, Width}

// scalastyle:off method.name

trait BoolFactory {
  /** Creates an empty Bool.
   */
  def apply(): Bool = new Bool()

  /** Creates Bool literal.
   */
  protected[chisel3] def Lit(x: Boolean): Bool = {
    val result = new Bool()
    val lit = ULit(if (x) 1 else 0, Width(1))
    // Ensure we have something capable of generating a name.
    lit.bindLitArg(result)
  }
}
