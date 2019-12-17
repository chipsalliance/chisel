// See LICENSE for license details.

package chiselTests.aop

import chisel3.testers.BasicTester
import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.aop.injecting.InjectingAspect

class AspectTester(results: Seq[Int]) extends BasicTester {
  val values = VecInit(results.map(_.U))
  val counter = RegInit(0.U(results.length.W))
  counter := counter + 1.U
  when(counter >= values.length.U) {
    stop()
  }.otherwise {
    when(reset.asBool() === false.B) {
      printf("values(%d) = %d\n", counter, values(counter))
      assert(counter === values(counter))
    }
  }
  val m = Module(new dummyModule)
}

class dummyModule() extends MultiIOModule {
  val io = IO(Output(UInt(3.W)))
  io := 3.U
}

class InjectionSpec extends ChiselFlatSpec {
  val correctValueAspect = InjectingAspect(
    {dut: AspectTester => Seq(dut)},
    {dut: AspectTester =>
      for(i <- 0 until dut.values.length) {
        dut.values(i) := i.U
      }
    }
  )

  val wrongValueAspect = InjectingAspect(
    {dut: AspectTester => Seq(dut)},
    {dut: AspectTester =>
      for(i <- 0 until dut.values.length) {
        dut.values(i) := (i + 1).U
      }
    }
  )

  // test to call IO()
  val ioAspect = InjectingAspect(
    {dut: AspectTester => Seq(dut.m)},
    {dut: dummyModule =>
    //for(i <- 0 until dut.values.length) {
    //  dut.values(i) := (i + 1).U
    //}
      val x = Wire(UInt(1.W))
      x := 1.U
      val dummy_out = chisel3.experimental.IO(Output(UInt(1.W))).suggestName("hello")
      dummy_out := dut.io
    }
  )

  "Test" should "compile and pass with correct aspect with injected IO" in {
    assertTesterPasses({ new AspectTester(Seq(9, 9, 9))} , Nil, Seq(ioAspect))
  }
  "Test" should "pass if inserted the correct values" in {
    assertTesterPasses{ new AspectTester(Seq(0, 1, 2)) }
  }
  "Test" should "fail if inserted the wrong values" in {
    assertTesterFails{ new AspectTester(Seq(9, 9, 9)) }
  }
  "Test" should "pass if pass wrong values, but correct with aspect" in {
    assertTesterPasses({ new AspectTester(Seq(9, 9, 9))} , Nil, Seq(correctValueAspect))
  }
  "Test" should "pass if pass wrong values, then wrong aspect, then correct aspect" in {
    assertTesterPasses({ new AspectTester(Seq(9, 9, 9))} , Nil, Seq(wrongValueAspect, correctValueAspect))
  }
  "Test" should "fail if pass wrong values, then correct aspect, then wrong aspect" in {
    assertTesterFails({ new AspectTester(Seq(9, 9, 9))} , Nil, Seq(correctValueAspect, wrongValueAspect))
  }
}
