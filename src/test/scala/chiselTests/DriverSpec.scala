// See LICENSE for license details.

package chiselTests

import chisel3._

import org.scalatest.{Matchers, FreeSpec}

class DummyModule extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(1.W))
    val out = Output(UInt(1.W))
  })
  io.out := io.in
}

class DriverSpec extends FreeSpec with Matchers {
  "Driver's execute methods are used to run chisel and firrtl" - {
    "options can be picked up from comand line with no args" in {
      Driver.execute(Array.empty[String], () => new DummyModule)
    }
    "options can be picked up from comand line setting top name" in {
      Driver.execute(Array("-tn", "dm", "-td", "local-build"), () => new DummyModule)
    }
    "execute returns a chisel execution result" in {
      val args = Array("--compiler", "low")
      val result = Driver.execute(Array.empty[String], () => new DummyModule)
      result shouldBe a[ChiselExecutionSucccess]
      val successResult = result.asInstanceOf[ChiselExecutionSucccess]
      successResult.emitted should include ("circuit DummyModule")
    }
  }
}
