// SPDX-License-Identifier: Apache-2.0

package chiselTests

import circt.stage.ChiselStage

import chisel3._
import chisel3.experimental.{annotate, AnyTargetable}
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.testing.scalatest.FileCheck
import chiselTests.experimental.hierarchy.Utils

import firrtl.transforms.{DontTouchAnnotation, NoDedupAnnotation}

import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class NewAnnotationsSpec extends AnyFreeSpec with Matchers with Utils with FileCheck {

  class MuchUsedModule extends Module {
    val io = IO(new Bundle {
      val in = Input(UInt(16.W))
      val out = Output(UInt(16.W))
    })
    io.out := io.in +% 1.U
  }

  class UsesMuchUsedModule extends Module {
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

    // Give two annotations as single element of the seq - ensures previous API works by wrapping into a seq.
    annotate(mod2)(Seq(new NoDedupAnnotation(mod2.toNamed)))
    annotate(mod3)(Seq(new NoDedupAnnotation(mod3.toNamed)))

    // Pass multiple annotations in the same seq - should get emitted out correctly.
    val ports = Seq(mod1.io.in, mod1.io.out)
    annotate(ports)({
      ports.map(p => new DontTouchAnnotation(p.toTarget))
    })
  }

  val stage = new ChiselStage
  "Ensure all annotations continue to be passed / digested correctly with the new API" - {
    "NoDedup and DontTouch work as expected" in {
      val dutAnnos = stage
        .execute(
          Array("--target", "chirrtl", "--target-dir", "test_run_dir"),
          Seq(ChiselGeneratorAnnotation(() => new UsesMuchUsedModule))
        )

      val dontTouchAnnos = dutAnnos.collect { case DontTouchAnnotation(target) => target.serialize }
      val noDedupAnnos = dutAnnos.collect { case NoDedupAnnotation(target) => target.serialize }
      require(dontTouchAnnos.size == 2, s"Exactly two DontTouch Annotations expected but got $dontTouchAnnos ")
      require(noDedupAnnos.size == 2, s"Exactly two NoDedup Annotations expected but got $noDedupAnnos ")
      val dontTouchAnnosCombined = dontTouchAnnos.mkString(",")
      val noDedupAnnosCombined = noDedupAnnos.mkString(",")

      noDedupAnnosCombined should include("~|MuchUsedModule_2")
      noDedupAnnosCombined should include("~|MuchUsedModule_3")
      dontTouchAnnosCombined should include("~|MuchUsedModule_1>io.out")
      dontTouchAnnosCombined should include("~|MuchUsedModule_1>io.in")
    }

    "It should be possible to annotate heterogeneous Targetable things" in {
      ChiselStage
        .emitCHIRRTL(new RawModule {
          override def desiredName: String = "Top"
          val in = IO(Input(UInt(8.W)))
          val out = IO(Output(UInt(8.W)))
          out := in
          // Given a Seq[UInt]
          val xs: Seq[UInt] = Seq(in, out)
          // We can manually use AnyTargetable to also include a Module
          // Using either type ascriptions to invoke the implicit conversion, or manually
          val ys = Seq[AnyTargetable](this) ++ xs.map(AnyTargetable(_))
          annotate(ys)(
            Seq(
              DontTouchAnnotation(in.toTarget),
              DontTouchAnnotation(out.toTarget),
              NoDedupAnnotation(this.toNamed)
            )
          )
        })
        .fileCheck()(
          """|CHECK:      "class":"firrtl.transforms.DontTouchAnnotation"
             |CHECK-NEXT: "target":"~|Top>in"
             |CHECK:      "class":"firrtl.transforms.DontTouchAnnotation"
             |CHECK-NEXT: "target":"~|Top>out"
             |CHECK:      "class":"firrtl.transforms.NoDedupAnnotation"
             |CHECK-NEXT: "target":"~|Top"
             |""".stripMargin
        )
    }
  }
}
