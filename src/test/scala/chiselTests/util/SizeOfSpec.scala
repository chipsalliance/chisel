package chiselTests.util

import circt.stage.CIRCTStage

import java.io.File
import java.io.{PrintWriter, Writer}

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.util.SizeOf

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

/** A test for intrinsics */
class SizeOfSpec extends AnyFlatSpec with Matchers {
  it should "Should work for types" in {
    val fir = circt.stage.ChiselStage.emitFIRRTLDialect(new SizeOfTop)
    val a = """firrtl.strictconnect %io_outw, %c65_ui32""".r
    (fir should include).regex(a)
  }
}
