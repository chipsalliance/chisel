// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.testers.BasicTester
import chisel3.stage.ChiselStage

class SinglePrintfTester() extends BasicTester {
  val x = 254.U
  printf("x=%x", x)
  stop()
}

class ASCIIPrintfTester() extends BasicTester {
  printf((0x20 to 0x7e).map(_.toChar).mkString.replace("%", "%%"))
  stop()
}

class MultiPrintfTester() extends BasicTester {
  val x = 254.U
  val y = 255.U
  printf("x=%x y=%x", x, y)
  stop()
}

class ASCIIPrintableTester extends BasicTester {
  printf(PString((0x20 to 0x7e).map(_.toChar).mkString("")))
  stop()
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

class PrintablePrintfScopeTester extends BasicTester {
  ChiselStage.elaborate {
    new Module {
      val mod = Module(new ScopeTesterModule)
      printf(mod.p)
    }
  }
  stop()
}

class PrintablePrintfWireScopeTester extends BasicTester {
  ChiselStage.elaborate {
    new Module {
      val mod = Module(new ScopeTesterModule)
      printf(mod.wp)
    }
  }
  stop()
}

class PrintfSpec extends ChiselFlatSpec {
  "A printf with a single argument" should "run" in {
    assertTesterPasses { new SinglePrintfTester }
  }
  "A printf with multiple arguments" should "run" in {
    assertTesterPasses { new MultiPrintfTester }
  }
  "A printf with ASCII characters 1-127" should "run" in {
    assertTesterPasses { new ASCIIPrintfTester }
  }
  "A printf with Printable ASCII characters 1-127" should "run" in {
    assertTesterPasses { new ASCIIPrintableTester }
  }
  "A printf with Printable" should "respect port scopes" in {
    assertTesterPasses { new PrintablePrintfScopeTester }
  }
  "A printf with Printable" should "respect wire scopes" in {
    a[ChiselException] should be thrownBy { assertTesterPasses { new PrintablePrintfWireScopeTester } }
  }
}
