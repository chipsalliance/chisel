// See LICENSE for license details.

package chiselTests

import chisel3._

class RebindingSpec extends ChiselFlatSpec {
  "Rebinding a literal" should "fail" in {
    a [chisel3.core.Binding.BindingException] should be thrownBy {
      elaborate { new Module {
        val io = IO(new Bundle {
          val a = 4.U
        })
      } }
    }
  }

  "Rebinding a hardware type" should "fail" in {
    a [chisel3.core.Binding.BindingException] should be thrownBy {
      elaborate { new Module {
        val io = IO(new Bundle {
          val a = Reg(UInt(32.W))
        })
      } }
    }
  }
}
