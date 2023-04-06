package chiselTests.util

import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import chisel3.testers.BasicTester
import chisel3.util.circt.SizeOf

import circt.stage.ChiselStage

import firrtl.stage.FirrtlCircuitAnnotation

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

class MyBundle extends Bundle {
  val a = UInt()
  val b = SInt()
}

class SizeOfTop extends Module {
  val io = IO(new Bundle {
    val w = Input(UInt(65.W))
    val x = Input(new MyBundle)
    val outw = UInt(32.W)
    val outx = UInt(32.W)
  })
  io.outw := SizeOf(io.w)
  io.outx := SizeOf(io.x)
}

/** A test for intrinsics.  Since chisel is producing intrinsics as tagged
  * extmodules (for now), we explicitly test the chirrtl and annotations rather
  * than the processed firrtl or verilog.  It is worth noting that annotations
  * are implemented (for now) in a way which makes the output valid for all
  * firrtl compilers, hence we write a localized, not end-to-end test
  */
class SizeOfSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitCHIRRTL(new SizeOfTop)
    val a1 = """extmodule SizeOf_0""".r
    (fir should include).regex(a1)
    val b1 = """defname = SizeOf_0""".r
    (fir should include).regex(b1)
    val a2 = """extmodule SizeOf_1""".r
    (fir should include).regex(a2)
    val b2 = """defname = SizeOf_1""".r
    (fir should include).regex(b2)

    // The second elaboration uses a unique name since the Builder is reused (?)
    val c = """Intrinsic\(~SizeOfTop\|SizeOf.*,circt.sizeof\)"""
    ((new ChiselStage)
      .execute(
        args = Array("--target", "chirrtl"),
        annotations = Seq(chisel3.stage.ChiselGeneratorAnnotation(() => new SizeOfTop))
      )
      .flatMap {
        case FirrtlCircuitAnnotation(circuit) => circuit.annotations
        case _                                => None
      }
      .mkString("\n") should include).regex(c)
  }
}
