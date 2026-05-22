// SPDX-License-Identifier: Apache-2.0
package chisel3.debug

import chisel3._
import chisel3.experimental.IntrinsicModule
import chisel3.experimental.hierarchy.Definition
import chisel3.internal.firrtl.ir
import chisel3.properties.{Class => PropertiesClass}
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation}
import chisel3.stage.phases.{AddDebugIntrinsics, Elaborate}
import circt.stage.ChiselStage
import firrtl.{annoSeqToSeq, seqToAnnoSeq, AnnotationSeq}
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

private class DebugSimpleModule extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))
  out := in
}

private class DebugParamModule(val width: Int, val hasReset: Boolean) extends Module {
  override def desiredName = s"DebugParamModule_w${width}_r$hasReset"
  val in = IO(Input(UInt(width.W)))
  val out = IO(Output(UInt(width.W)))
  if (hasReset) { val reg = RegInit(0.U(width.W)); reg := in; out := reg }
  else out := in
}

private class DebugSubModule extends Module {
  val a = IO(Input(UInt(4.W)))
  val b = IO(Output(UInt(4.W)))
  b := a + 1.U
}

private class DebugTopModule extends Module {
  val in = IO(Input(UInt(4.W)))
  val out = IO(Output(UInt(4.W)))
  val sub = Module(new DebugSubModule)
  sub.a := in
  out := sub.b
}

private class DebugParamBundle(val n: Int) extends Bundle {
  val data = UInt(n.W)
}

private class DebugBundleModule(val width: Int) extends Module {
  val io = IO(new DebugParamBundle(width))
}

private class DebugMyIntrinsicMod extends IntrinsicModule("my_test_intr") {
  val o = IO(Output(Bool()))
}

private class DebugWithIntrinsic extends RawModule {
  val intr = Module(new DebugMyIntrinsicMod)
}

private class DebugMyClass extends PropertiesClass

private class DebugWithClass extends RawModule {
  val descDef = Definition(new DebugMyClass)
}

class DebugIntrinsicsSpec extends AnyFunSpec with Matchers {
  import DebugTestUtils._

  describe("circt_debug_moduleinfo") {

    it("emits exactly one moduleinfo per module") {
      val chirrtl = emit(new DebugSimpleModule)
      countOccurrences(chirrtl, moduleInfoPattern("DebugSimpleModule")) should be(1)
    }

    it("emits moduleinfo with typeName for a parametrized module") {
      val chirrtl = emit(new DebugParamModule(8, false))
      countOccurrences(chirrtl, moduleInfoPattern("DebugParamModule_w8_rfalse")) should be(1)
    }

    it("emits moduleinfo for every module in a hierarchy") {
      val chirrtl = emit(new DebugTopModule)
      countOccurrences(chirrtl, moduleInfoPattern("DebugTopModule")) should be(1)
      countOccurrences(chirrtl, moduleInfoPattern("DebugSubModule")) should be(1)
    }

    it("emits moduleinfo with constructor params serialized in params field") {
      val chirrtl = emit(new DebugParamModule(5, true))
      val name = "DebugParamModule_w5_rtrue"
      countOccurrences(chirrtl, moduleInfoPattern(name)) should be(1)
      val params = requireParams(chirrtl, name)
      assertParam(params, "width", "Int", 5) // numbers serialize as JSON strings
      assertParam(params, "hasReset", "Boolean", true) // Booleans stay as native JSON bool
    }

    it("does not emit moduleinfo for blackboxes") {
      import DebugTestCircuits.ModuleCircuits._
      val chirrtl = emit(new TopCircuitBlackBox)
      countOccurrences(chirrtl, moduleInfoPattern("TopCircuitBlackBox")) should be(1)
      countOccurrences(chirrtl, moduleInfoPattern("MyBlackBox")) should be(0)
    }

    it("does not emit moduleinfo for IntrinsicModule") {
      val chirrtl = emit(new DebugWithIntrinsic)
      countOccurrences(chirrtl, moduleInfoPattern("DebugWithIntrinsic")) should be(1)
      countOccurrences(chirrtl, moduleInfoPattern("DebugMyIntrinsicMod")) should be(0)
    }

    it("emits moduleinfo for chisel3.properties.Class (DefClass)") {
      val chirrtl = emit(new DebugWithClass)
      countOccurrences(chirrtl, moduleInfoPattern("DebugWithClass")) should be(1)
      countOccurrences(chirrtl, moduleInfoPattern("DebugMyClass")) should be(1)
    }

    it("emits moduleinfo with params for a module whose port is a parametrized bundle") {
      val chirrtl = emit(new DebugBundleModule(16))
      countOccurrences(chirrtl, moduleInfoPattern("DebugBundleModule")) should be(1)
      val params = requireParams(chirrtl, "DebugBundleModule")
      assertParam(params, "width", "Int", 16)
    }
  }

  describe("EmitDebugIntrinsicsAnnotation toggle") {
    it("emits no circt_debug_* intrinsics by default") {
      val chirrtl = ChiselStage.emitCHIRRTL(new DebugSimpleModule)
      (chirrtl should not).include("circt_debug_")
    }

    it("emits circt_debug_* intrinsics with --with-experimental-debug-intrinsics") {
      val chirrtl = ChiselStage.emitCHIRRTL(new DebugSimpleModule, args = Array("--with-experimental-debug-intrinsics"))
      chirrtl should include("circt_debug_moduleinfo")
      chirrtl should include("circt_debug_var")
    }
  }

  describe("AddDebugIntrinsics phase idempotency") {
    // Walk the in-memory circuit (rather than re-emitting CHIRRTL) so we count
    // exactly the secret commands the phase added -- this is what would double
    // if the consume-the-annotation guard regressed.
    def countDebugIntrinsics(circuit: ir.Circuit): Int = {
      def inBlock(block: ir.Block): Int = {
        val direct = (block.getCommands() ++ block.getSecretCommands()).count {
          case d: ir.DefIntrinsic => d.intrinsic.startsWith("circt_debug_")
          case _ => false
        }
        val nested = block
          .getCommands()
          .map {
            case w:  ir.When        => inBlock(w.ifRegion) + (if (w.hasElse) inBlock(w.elseRegion) else 0)
            case lb: ir.LayerBlock  => inBlock(lb.region)
            case dc: ir.DefContract => inBlock(dc.region)
            case _ => 0
          }
          .sum
        direct + nested
      }
      circuit.components.map {
        case dm: ir.DefModule => inBlock(dm.block)
        case dc: ir.DefClass  => inBlock(dc.block)
        case _ => 0
      }.sum
    }

    it("consumes EmitDebugIntrinsicsAnnotation so a second pass does not double-emit") {
      val elaborated: AnnotationSeq = (new Elaborate)
        .transform(Seq(ChiselGeneratorAnnotation(() => new DebugTopModule)))
      val withFlag = elaborated :+ EmitDebugIntrinsicsAnnotation
      val pass = new AddDebugIntrinsics

      val afterFirst = pass.transform(withFlag)
      afterFirst.exists(_ == EmitDebugIntrinsicsAnnotation) shouldBe false

      val cca1 = afterFirst.collectFirst { case a: ChiselCircuitAnnotation => a }
        .getOrElse(fail("no ChiselCircuitAnnotation after first pass"))
      val countOnce = countDebugIntrinsics(cca1.elaboratedCircuit._circuit)
      countOnce should be > 0

      val afterSecond = pass.transform(afterFirst)
      val cca2 = afterSecond.collectFirst { case a: ChiselCircuitAnnotation => a }
        .getOrElse(fail("no ChiselCircuitAnnotation after second pass"))
      countDebugIntrinsics(cca2.elaboratedCircuit._circuit) shouldEqual countOnce
    }
  }
}
