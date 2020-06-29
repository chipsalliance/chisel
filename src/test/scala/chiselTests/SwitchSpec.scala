// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util._

class SwitchSpec extends ChiselFlatSpec with Utils {
  "switch" should "require literal conditions" in {
    a [java.lang.IllegalArgumentException] should be thrownBy extractCause[IllegalArgumentException] {
      ChiselStage.elaborate(new Module {
        val io = IO(new Bundle {})
        val state = RegInit(0.U)
        val wire = WireDefault(0.U)
        switch (state) {
          is (wire) { state := 1.U }
        }
      })
    }
  }
  it should "require mutually exclusive conditions" in {
    a [java.lang.IllegalArgumentException] should be thrownBy extractCause[IllegalArgumentException] {
      ChiselStage.elaborate(new Module {
        val io = IO(new Bundle {})
        val state = RegInit(0.U)
        switch (state) {
          is (0.U) { state := 1.U }
          is (1.U) { state := 2.U }
          is (0.U) { state := 3.U }
        }
      })
    }
  }
}
