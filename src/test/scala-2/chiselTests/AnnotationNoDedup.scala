// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.doNotDedup
import circt.stage.ChiselStage
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class MuchUsedModule extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })
  io.out := io.in +% 1.U
}

class UsesMuchUsedModule(addAnnos: Boolean) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  val mod0 = Module(new MuchUsedModule)
  val mod1 = Module(new MuchUsedModule)
  val mod2 = Module(new MuchUsedModule)
  val mod3 = Module(new MuchUsedModule)

  mod0.io.in := io.in
  mod1.io.in := mod0.io.out
  mod2.io.in := mod1.io.out
  mod3.io.in := mod2.io.out
  io.out := mod3.io.out

  if (addAnnos) {
    doNotDedup(mod1)
    doNotDedup(mod3)
  }
}

class AnnotationNoDedup extends AnyFreeSpec with Matchers {
  "Firrtl provides transform that reduces identical modules to a single instance" - {
    "Annotations can be added which will prevent this deduplication for specific modules instances" in {
      val verilog = ChiselStage.emitSystemVerilog(new UsesMuchUsedModule(addAnnos = true))
      verilog should include("module MuchUsedModule(")
      verilog should include("module MuchUsedModule_1(")
      verilog should include("module MuchUsedModule_3(")
      (verilog should not).include("module MuchUsedModule_2(")
      (verilog should not).include("module MuchUsedModule_4(")
    }
    "Turning off these annotations dedups all the occurrences" in {
      val verilog = ChiselStage.emitSystemVerilog(new UsesMuchUsedModule(addAnnos = false))
      verilog should include("module MuchUsedModule(")
      (verilog should not).include("module MuchUsedModule_1(")
      (verilog should not).include("module MuchUsedModule_3(")
      (verilog should not).include("module MuchUsedModule_2(")
      (verilog should not).include("module MuchUsedModule_4(")
    }
  }
}
