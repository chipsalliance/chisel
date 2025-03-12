// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SinglePrintfTester() extends Module {
  val x = 254.U
  printf("x=%x", x)
}

class ASCIIPrintfTester() extends Module {
  printf((0x20 to 0x7e).map(_.toChar).mkString.replace("%", "%%"))
}

class MultiPrintfTester() extends Module {
  val x = 254.U
  val y = 255.U
  printf("x=%x y=%x", x, y)
}

class ASCIIPrintableTester extends Module {
  printf(PString((0x20 to 0x7e).map(_.toChar).mkString("")))
}

class ScopeTesterModule extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))
  out := in

  val w = Wire(UInt(8.W))
  w := 125.U

  val p = cf"$in"
  val wp = cf"$w"
}

class PrintfSpec extends AnyFlatSpec with Matchers {
  "A printf with a single argument" should "elaborate" in {
    val chirrtl = ChiselStage.emitCHIRRTL(new SinglePrintfTester)

    info("printfs are wrapped in the `Verification` layerblock by default")
    chirrtl should include("layerblock Verification")
  }
  "A printf with multiple arguments" should "elaborate" in {
    ChiselStage.emitCHIRRTL(new MultiPrintfTester)
  }
  "A printf with ASCII characters 1-127" should "elaborate" in {
    ChiselStage.emitCHIRRTL(new ASCIIPrintfTester)
  }
  "A printf with Printable ASCII characters 1-127" should "elaborate" in {
    ChiselStage.emitCHIRRTL(new ASCIIPrintableTester)
  }
  "A printf with Printable targeting a format string using a port in another module" should "elaborate" in {
    ChiselStage.emitCHIRRTL {
      new Module {
        val mod = Module(new ScopeTesterModule)
        printf(mod.p)
      }
    }
  }
  "A printf with Printable targeting a format string using a wire inside another module" should "error" in {
    a[ChiselException] should be thrownBy {
      circt.stage.ChiselStage.emitCHIRRTL {
        new Module {
          val mod = Module(new ScopeTesterModule)
          printf(mod.wp)
        }
      }
    }
  }
}
