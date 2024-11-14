// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import org.scalatest.matchers.should.Matchers
import chiselTests.{ChiselFlatSpec, MatchesAndOmits}
import chisel3.util.experimental.{InlineInstance, InlineInstanceAllowDedup}

class InlineInstanceSpec extends ChiselFlatSpec with MatchesAndOmits {
  class ModuleA extends RawModule {
    val w = dontTouch(WireInit(false.B))
  }

  class ModuleB extends RawModule with InlineInstance {
    val w = dontTouch(WireInit(false.B))
  }

  class TopModule extends RawModule {
    val a = Module(new ModuleA)
    val b = Module(new ModuleB)
  }

  "InlineInstanceAllowDedup" should "Inline any module that dedups with a module marked inline" in {
    val verilog = ChiselStage.emitSystemVerilog(new TopModule)
    matchesAndOmits(verilog)(
      "module TopModule()",
      "module ModuleA();"
    )(
      "module ModuleB()"
    )
  }
}

class InlineInstanceAllowDedupSpec extends ChiselFlatSpec with MatchesAndOmits {
  class ModuleA extends RawModule {
    val w = dontTouch(WireInit(false.B))
  }

  class ModuleB extends RawModule with InlineInstanceAllowDedup {
    val w = dontTouch(WireInit(false.B))
  }

  class TopModule extends RawModule {
    val a = Module(new ModuleA)
    val b = Module(new ModuleB)
  }

  "InlineInstanceAllowDedup" should "Inline any module that dedups with a module marked inline" in {
    val verilog = ChiselStage.emitSystemVerilog(new TopModule)
    matchesAndOmits(verilog)(
      "module TopModule()"
    )(
      "module ModuleA()",
      "module ModuleB()"
    )
  }
}
