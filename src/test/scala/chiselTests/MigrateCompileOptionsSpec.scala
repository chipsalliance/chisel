// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3.stage.ChiselStage
import chisel3.ImplicitInvalidate
import chisel3.ExplicitCompileOptions

import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks

object MigrationExamples {
  object InferResets {
    import Chisel.{defaultCompileOptions => _, _}
    import chisel3.RequireSyncReset
    implicit val migrateIR = new chisel3.CompileOptions {
      val connectFieldsMustMatch = false
      val declaredTypeMustBeUnbound = false
      val dontTryConnectionsSwapped = false
      val dontAssumeDirectionality = false
      val checkSynthesizable = false
      val explicitInvalidate = false
      val inferModuleReset = false

      override val migrateInferModuleReset = true
    }

    class Foo extends Module {
      val io = new Bundle {}
    }
    class FooWithRequireSyncReset extends Module with RequireSyncReset {
      val io = new Bundle {}
    }
  }
  object ExplicitInvalidate {
    import chisel3.ImplicitInvalidate
    val migrateEI = new chisel3.CompileOptions {
      val connectFieldsMustMatch = false
      val declaredTypeMustBeUnbound = false
      val dontTryConnectionsSwapped = false
      val dontAssumeDirectionality = false
      val checkSynthesizable = false
      val explicitInvalidate = true
      val inferModuleReset = false
    }
    object ChiselChildren {
      import Chisel.{defaultCompileOptions => _, _}
      implicit val options = migrateEI
      class Foo extends Module {
        val io = new Bundle {
          val out = Output(UInt(width = 3))
        }
      }
      class FooWithImplicitInvalidate extends Module with ImplicitInvalidate {
        val io = new Bundle {
          val out = Output(UInt(width = 3))
        }
      }
      class FooWire extends Module {
        val io = new Bundle {}
        val wire = Wire(Bool())
      }
      class FooWireWithImplicitInvalidate extends Module with ImplicitInvalidate {
        val io = new Bundle {}
        val wire = Wire(Bool())
      }
    }
    object chisel3Children {
      import chisel3._
      class Foo extends Module {
        val in = IO(chisel3.Input(UInt(3.W)))
      }
    }
    object ChiselParents {
      import Chisel.{defaultCompileOptions => _, _}
      implicit val options = migrateEI

      class FooParent extends Module {
        val io = new Bundle {}
        val i = Module(new chisel3Children.Foo)
      }
      class FooParentWithImplicitInvalidate extends Module with ImplicitInvalidate {
        val io = new Bundle {}
        val i = Module(new chisel3Children.Foo)
      }
    }
  }
}

class MigrateCompileOptionsSpec extends ChiselFunSpec with Utils {
  import Chisel.{defaultCompileOptions => _, _}
  import chisel3.RequireSyncReset

  describe("(0): Migrating infer resets") {
    import MigrationExamples.InferResets._
    it("(0.a): Error if migrating, but not extended RequireSyncReset") {
      intercept[Exception] { ChiselStage.elaborate(new Foo) }
    }
    it("(0.b): Not error if migrating, and you mix with RequireSyncReset") {
      ChiselStage.elaborate(new FooWithRequireSyncReset)
    }
  }

  describe("(1): Migrating explicit invalidate") {
    import MigrationExamples.ExplicitInvalidate._

    it("(1.a): error if migrating module input, but not extending ImplicitInvalidate") {
      intercept[_root_.firrtl.passes.CheckInitialization.RefNotInitializedException] {
        ChiselStage.emitVerilog(new ChiselChildren.Foo)
      }
    }
    it("(1.b): succeed if migrating module input with extending ImplicitInvalidate") {
      ChiselStage.emitVerilog(new ChiselChildren.FooWithImplicitInvalidate)
    }

    it("(1.c): error if migrating instance output, but not extending ImplicitInvalidate") {
      intercept[_root_.firrtl.passes.CheckInitialization.RefNotInitializedException] {
        ChiselStage.emitVerilog(new ChiselParents.FooParent)
      }
    }
    it("(1.d): succeed if migrating instance output with extending ImplicitInvalidate") {
      ChiselStage.emitVerilog(new ChiselParents.FooParentWithImplicitInvalidate)
    }

    it("(1.e): error if migrating wire declaration, but not extending ImplicitInvalidate") {
      intercept[_root_.firrtl.passes.CheckInitialization.RefNotInitializedException] {
        ChiselStage.emitVerilog(new ChiselChildren.FooWire)
      }
    }
    it("(1.f): succeed if migrating wire declaration with extending ImplicitInvalidate") {
      ChiselStage.emitVerilog(new ChiselChildren.FooWireWithImplicitInvalidate)
    }
  }
}
