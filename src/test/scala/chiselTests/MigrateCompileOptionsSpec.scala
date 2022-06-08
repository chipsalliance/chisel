// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3.stage.ChiselStage

import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks

class MigrateCompileOptionsSpec extends ChiselFlatSpec with ScalaCheckDrivenPropertyChecks with Utils {
  import Chisel.{defaultCompileOptions => _, _}
  import chisel3.RequireSyncReset

  behavior.of("Migrating infer resets")

  val migrateIR = new chisel3.CompileOptions {
    val connectFieldsMustMatch = false
    val declaredTypeMustBeUnbound = false
    val dontTryConnectionsSwapped = false
    val dontAssumeDirectionality = false
    val checkSynthesizable = false
    val explicitInvalidate = false
    val inferModuleReset = false

    override val migrateInferModuleReset = true
  }

  it should "error if migrating, but not extended RequireSyncReset" in {
    implicit val options = migrateIR
    class Foo extends Module {
      val io = new Bundle {}
    }
    intercept[Exception] {
      ChiselStage.elaborate(new Foo)
    }
  }
  it should "not error if migrating, and you mix with RequireSyncReset" in {
    implicit val options = migrateIR
    class Foo extends Module with RequireSyncReset {
      val io = new Bundle {}
    }
    ChiselStage.elaborate(new Foo)
  }
}
