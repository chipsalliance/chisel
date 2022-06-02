// SPDX-License-Identifier: Apache-2.0

package chiselTests.naming

import chisel3._
import chisel3.aop.Select
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.experimental.{dump, noPrefix, prefix, treedump}
import chiselTests.{ChiselPropSpec, Utils}
import chisel3.stage.{DesignAnnotation, NoRunFirrtlCompilerAnnotation}
import firrtl.options.{Dependency, Phase, PhaseManager}
import firrtl.options.phases.DeletedWrapper

class SuggestNameSpec extends ChiselPropSpec with Utils {
  implicit val minimumMajorVersion: Int = 12

  property("0. Calling suggestName 2x should be a runtime deprecation") {
    class Test extends Module {
      {
        val wire = {
          val x = WireInit(0.U(3.W)).suggestName("mywire")
          dontTouch(x)
        }
        wire.suggestName("somethingElse")
      }
    }
    val (log, _) = grabLog(ChiselStage.emitVerilog(new Test()))
    log should include(
      "Calling suggestName (somethingElse, when already called with Some(mywire)) will become an error in Chisel 3.6"
    )
  }

  property("1. Calling suggestName outside of a Builder context should be an error") {
    class Test extends Module {
      val wire = {
        val x = WireInit(0.U(3.W))
        dontTouch(x)
      }
    }

    val pm = new PhaseManager(Seq(Dependency[chisel3.stage.phases.Checks], Dependency[chisel3.stage.phases.Elaborate]))
    val test = pm
      .transform(Seq(ChiselGeneratorAnnotation(() => new Test()), NoRunFirrtlCompilerAnnotation))
      .collectFirst {
        case d: DesignAnnotation[_] => d
      }
      .get
      .design
    val caught = intercept[IllegalArgumentException] {
      test.asInstanceOf[Test].wire.suggestName("somethingElse")
    }
    caught.getMessage should include("suggestName (somethingElse) should only be called from a Builder context")
  }

  property("2. Calling suggestName after module close should be a runtime deprecation") {
    class Child extends Module {
      val wire = {
        val x = WireInit(0.U(3.W))
        dontTouch(x)
      }
    }
    class Test extends Module {
      val child = Module(new Child())
      child.wire.suggestName("somethingElse")
    }
    val (log, _) = grabLog(ChiselStage.emitVerilog(new Test()))
    log should include(
      "Calling suggestName (somethingElse, when the name was already computed "
    )
  }

  property("3. Calling suggestName after toString should be a runtime deprecation") {
    class Test extends Module {
      val wire = {
        val x = WireInit(0.U(3.W))
        val y = x.toString
        x.suggestName("somethingElse")
      }
    }
    val (log, _) = grabLog(ChiselStage.emitVerilog(new Test()))
    log should include(
      "Calling suggestName (somethingElse, when the name was already computed "
    )
  }

  property("4. Calling suggestName with the same thing prefix would have given should be a runtime deprecation") {
    class Test extends Module {
      val wire = {
        val x = WireInit(0.U(3.W)).suggestName("wire")
        dontTouch(x)
      }
    }
    val (log, _) = grabLog(ChiselStage.emitVerilog(new Test()))
    log should include(
      "calling suggestName(wire) had no effect as it is the same as the auto prefixed name"
    )
  }
}
