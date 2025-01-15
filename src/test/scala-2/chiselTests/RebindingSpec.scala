// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

class RebindingSpec extends ChiselFlatSpec with Utils {
  "Rebinding a literal" should "fail" in {
    a[BindingException] should be thrownBy extractCause[BindingException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          val io = IO(new Bundle {
            val a = 4.U
          })
        }
      }
    }
  }

  "Rebinding a hardware type" should "fail" in {
    a[BindingException] should be thrownBy extractCause[BindingException] {
      ChiselStage.emitCHIRRTL {
        new Module {
          val io = IO(new Bundle {
            val a = Reg(UInt(32.W))
          })
        }
      }
    }
  }
}
