// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage

class RebindingSpec extends ChiselFlatSpec with Utils {
  "Rebinding a literal" should "fail" in {
    a [BindingException] should be thrownBy extractCause[BindingException] {
      ChiselStage.elaborate { new Module {
        val io = IO(new Bundle {
          val a = 4.U
        })
      } }
    }
  }

  "Rebinding a hardware type" should "fail" in {
    a [BindingException] should be thrownBy extractCause[BindingException] {
      ChiselStage.elaborate { new Module {
        val io = IO(new Bundle {
          val a = Reg(UInt(32.W))
        })
      } }
    }
  }
}
