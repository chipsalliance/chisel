// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.experimental.SourceInfo

object PrintfMacrosCompat {
  private[chisel3] def printfWithReset(
    pable: Printable
  )(
    using sourceInfo: SourceInfo
  ): chisel3.printf.Printf = {
    var printfId: chisel3.printf.Printf = null
    when(!Module.reset.asBool) {
      printfId = printfWithoutReset(pable)
    }
    printfId
  }

  private[chisel3] def printfWithoutReset(
    pable: Printable
  )(
    using sourceInfo: SourceInfo
  ): chisel3.printf.Printf = {
    val clock = Builder.forcedClock
    val printfId = new chisel3.printf.Printf(pable)

    Printable.checkScope(pable)

    pushCommand(chisel3.internal.firrtl.ir.Printf(printfId, sourceInfo, clock.ref, pable))
    printfId
  }
}
