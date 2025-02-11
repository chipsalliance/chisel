// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.{is, switch}
import circt.stage.ChiselStage

class SwitchSpec extends ChiselFlatSpec with Utils {
  "switch" should "require literal conditions" in {
    a[java.lang.IllegalArgumentException] should be thrownBy extractCause[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        val state = RegInit(0.U)
        val wire = WireDefault(0.U)
        switch(state) {
          is(wire) { state := 1.U }
        }
      })
    }
  }
  it should "require mutually exclusive conditions" in {
    a[java.lang.IllegalArgumentException] should be thrownBy extractCause[IllegalArgumentException] {
      ChiselStage.emitCHIRRTL(new Module {
        val io = IO(new Bundle {})
        val state = RegInit(0.U)
        switch(state) {
          is(0.U) { state := 1.U }
          is(1.U) { state := 2.U }
          is(0.U) { state := 3.U }
        }
      })
    }
  }
  it should "provide useful source locators" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val in = Input(UInt(2.W))
        val out = Output(UInt(2.W))
      })

      io.out := 0.U
      switch(io.in) {
        is(0.U) { io.out := 3.U }
        is(1.U) { io.out := 0.U }
        is(2.U) { io.out := 1.U }
        is(3.U) { io.out := 3.U }
      }
    })

    (chirrtl should not).include("Conditional.scala")
  }
}
