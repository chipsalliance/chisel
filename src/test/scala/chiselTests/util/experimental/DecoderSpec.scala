package chiselTests.util.experimental

import chisel3._
import chiselTests._
import chisel3.testers.BasicTester
import chisel3.stage.ChiselStage
import chisel3.util.BitPat
import chisel3.util.experimental.decode.{TruthTable, decodeAs}
// import org.scalatest.flatspec.AnyFlatSpec
class OutputBundle extends Bundle {
  val s1 = UInt(2.W)
  val s2 = UInt(2.W)
  val s3 = Bool()
}
class DecodeAs extends Module {
    val io = IO(new Bundle {
        val i = Input(UInt(3.W))
        val o = Output(new OutputBundle())
    })

    val signals = decodeAs(
        (new OutputBundle),
        io.i,
        TruthTable(
            Array(
            BitPat("b001")  -> BitPat(1.U(2.W)) ## BitPat(2.U(2.W)) ## BitPat.Y(),
            BitPat("b?11")  -> BitPat(2.U(2.W)) ## BitPat(3.U(2.W)) ## BitPat.Y(),
            ),                 BitPat(0.U(2.W)) ## BitPat(0.U(2.W)) ## BitPat.N() // Default values
        )
    )
    io.o := signals
}

class DecodeAsTester(i: String, o1: Int, o2: Int, o3: Boolean) extends BasicTester {
  val dut = Module(new DecodeAs())
  dut.io.i := i.U

  assert(dut.io.o.s1 === o1.U)
  assert(dut.io.o.s2 === o2.U)
  assert(dut.io.o.s3 === o3.B)
  stop()
}

class DecoderSpec extends ChiselPropSpec {
    property("decoder should decodeAs to an existing bundle") {
        assertTesterPasses{ new DecodeAsTester("b001", 1, 2, true) }
    }

    property("decoder should decodeAs to an existing bundle with default values") {
        assertTesterPasses{ new DecodeAsTester("b101", 0, 0, false) }
    }
}
