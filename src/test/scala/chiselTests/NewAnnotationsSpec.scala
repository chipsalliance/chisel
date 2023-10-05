package chiselTests
import chisel3._
import chisel3.experimental.{annotate, ChiselMultiAnnotation}
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.ChiselStage
import firrtl.stage.FirrtlCircuitAnnotation
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers
import firrtl.transforms.NoDedupAnnotation
import firrtl.transforms.DontTouchAnnotation

class NewAnnotationsSpec extends AnyFreeSpec with Matchers {

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
    annotate(new ChiselMultiAnnotation { def toFirrtl = Seq(new NoDedupAnnotation(mod2.toNamed)) })
    annotate(new ChiselMultiAnnotation { def toFirrtl = Seq(new NoDedupAnnotation(mod3.toNamed)) })

    // Pass multiple annotations in the same seq - should get emitted out correctly.
    annotate(new ChiselMultiAnnotation {
      def toFirrtl =
        Seq(new DontTouchAnnotation(mod1.io.in.toNamed), new DontTouchAnnotation(mod1.io.out.toNamed))
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

      noDedupAnnosCombined should include("~UsesMuchUsedModule|MuchUsedModule_2")
      noDedupAnnosCombined should include("~UsesMuchUsedModule|MuchUsedModule_3")
      dontTouchAnnosCombined should include("~UsesMuchUsedModule|MuchUsedModule_1>io.out")
      dontTouchAnnosCombined should include("~UsesMuchUsedModule|MuchUsedModule_1>io.in")

    }
  }
}
