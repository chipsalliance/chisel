package chisel3.tests

import org.scalatest._

import chisel3._
import chisel3.testers2._

class ElementTest extends FlatSpec with ChiselScalatestTester {
  behavior of "Testers2 with Element types"

  // TODO: automatically detect overflow conditions and error out

  it should "work with UInt" in {
    test(new Module {
      val io = IO(new Bundle {
        val in1 = Input(UInt(8.W))
        val in2 = Input(UInt(8.W))
        val out = Output(UInt(8.W))

        def expect(in1Val: UInt, in2Val: UInt, outVal: UInt) {
          in1.poke(in1Val)
          in2.poke(in2Val)
          out.expect(outVal)
        }
      })
      io.out := io.in1 + io.in2
    }) { c =>
      c.io.expect(0.U, 0.U, 0.U)
      c.io.expect(1.U, 0.U, 1.U)
      c.io.expect(0.U, 1.U, 1.U)
      c.io.expect(1.U, 1.U, 2.U)
      c.io.expect(254.U, 1.U, 255.U)
      c.io.expect(255.U, 1.U, 0.U)  // overflow behavior
      c.io.expect(255.U, 255.U, 254.U)  // overflow behavior
    }
  }

  it should "work with SInt" in {
    test(new Module {
      val io = IO(new Bundle {
        val in1 = Input(SInt(8.W))
        val in2 = Input(SInt(8.W))
        val out = Output(SInt(8.W))

        def expect(in1Val: SInt, in2Val: SInt, outVal: SInt) {
          in1.poke(in1Val)
          in2.poke(in2Val)
          out.expect(outVal)
        }
      })
      io.out := io.in1 + io.in2
    }) { c =>
      c.io.expect(0.S, 0.S, 0.S)
      c.io.expect(1.S, 0.S, 1.S)
      c.io.expect(0.S, 1.S, 1.S)
      c.io.expect(1.S, 1.S, 2.S)

      c.io.expect(127.S, -1.S, 126.S)
      c.io.expect(127.S, -127.S, 0.S)
      c.io.expect(-128.S, 127.S, -1.S)

      c.io.expect(127.S, 127.S, -2.S)
    }
  }
}
