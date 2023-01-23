package chiselTests.util

import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.testers.BasicTester
import chisel3.util.circt.SizeOf

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.io.Source

class SizeOfTop extends Module {
  val io = IO(new Bundle{
    val w = Input(UInt(65.W))
    val outw = UInt(32.W)
  })
  io.outw  := SizeOf(io.w)
}

/** A test for intrinsics.  Since chisel is producing intrinsics as tagged 
 * extmodules (for now), we explicitly test the chirrtl and annotations rather
 * than the processed firrtl or verilog.  It is worth noting that annotations 
 * are implemented (for now) in a way which makes the output valid for all
 * firrtl compilers, hence we write a localized, not end-to-end test */
class SizeOfSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = ChiselStage.emitChirrtl(new SizeOfTop)
    val a = """extmodule SizeOf8""".r
    (fir should include).regex(a)
    val b = """defname = SizeOf8""".r
    (fir should include).regex(b)

    // The second elaboration uses a unique name since the Builder is reused (?)
    val anno = ChiselStage.emitAnnotations(new SizeOfTop)
    val c = """Intrinsic\(~SizeOfTop\|SizeOf.*,circt.sizeof\)"""
    (anno should include).regex(c)
  }
}
