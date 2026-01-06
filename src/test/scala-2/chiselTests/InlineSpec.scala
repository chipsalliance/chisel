// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.dedupGroup
import chisel3.testing.FileCheck
import chisel3.util.experimental.{InlineInstance, InlineInstanceAllowDedup}
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class InlineInstanceSpec extends AnyFlatSpec with Matchers with FileCheck {
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
    ChiselStage
      .emitSystemVerilog(new TopModule)
      .fileCheck("--implicit-check-not=ModuleB")(
        """|CHECK: ModuleA()
           |CHECK: TopModule()
           |""".stripMargin
      )
  }
}

class InlineInstanceAllowDedupSpec extends AnyFlatSpec with Matchers with FileCheck {
  class ModuleA extends RawModule {
    val w = dontTouch(WireInit(false.B))
  }

  class ModuleB extends RawModule with InlineInstanceAllowDedup {
    val w = dontTouch(WireInit(false.B))
  }

  class TopModule extends RawModule {
    val a = Module(new ModuleA)
    val b = Module(new ModuleB)
    Seq(a, b).foreach(dedupGroup(_, "TopModule"))
  }

  "InlineInstanceAllowDedup" should "Inline any module that dedups with a module marked inline" in {
    ChiselStage
      .emitSystemVerilog(new TopModule)
      .fileCheck()(
        """|CHECK-NOT: Module{{A|B}}
           |CHECK:     TopModule()
           |""".stripMargin
      )
  }
}
