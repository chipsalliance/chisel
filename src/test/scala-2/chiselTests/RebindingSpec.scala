// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RebindingSpec extends AnyFlatSpec with Matchers {
  "Rebinding a literal" should "fail" in {
    a[BindingException] should be thrownBy {
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
    a[BindingException] should be thrownBy {
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
