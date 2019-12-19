// See LICENSE for license details.

package chiselTests.aop

import chisel3.testers.BasicTester
import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.aop.injecting.InjectingAspect

class CounterAspectTester(results: Seq[Int]) extends BasicTester {
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
}

class IOAspectTester(value: UInt) extends BasicTester {
  val x = Wire(UInt(3.W))
  assert(x === value)
  stop()
}

class InjectionSpec extends ChiselFlatSpec {
  val correctValueAspect = InjectingAspect(
    {dut: CounterAspectTester => Seq(dut)},
    {dut: CounterAspectTester =>
      for(i <- 0 until dut.values.length) {
        dut.values(i) := i.U
      }
    }
  )

  val wrongValueAspect = InjectingAspect(
    {dut: CounterAspectTester => Seq(dut)},
    {dut: CounterAspectTester =>
      for(i <- 0 until dut.values.length) {
        dut.values(i) := (i + 1).U
      }
    }
  )

  val ioAspect = InjectingAspect(
    {dut: IOAspectTester => Seq(dut)},
    {dut: IOAspectTester =>
      val dummy_out = chisel3.experimental.IO(Output(UInt(3.W))).suggestName("hello")
      dummy_out := 5.U
      dut.x := dummy_out
      dontTouch(dummy_out) // test if dontTouch() works for IO
    }
  )

  "Test" should "pass if inserted the correct values" in {
    assertTesterPasses{ new CounterAspectTester(Seq(0, 1, 2)) }
  }
  "Test" should "fail if inserted the wrong values" in {
    assertTesterFails{ new CounterAspectTester(Seq(9, 9, 9)) }
  }
  "Test" should "pass if pass wrong values, but correct with aspect" in {
    assertTesterPasses({ new CounterAspectTester(Seq(9, 9, 9))} , Nil, Seq(correctValueAspect))
  }
  "Test" should "pass if pass wrong values, then wrong aspect, then correct aspect" in {
    assertTesterPasses({ new CounterAspectTester(Seq(9, 9, 9))} , Nil, Seq(wrongValueAspect, correctValueAspect))
  }
  "Test" should "fail if pass wrong values, then correct aspect, then wrong aspect" in {
    assertTesterFails({new CounterAspectTester(Seq(9, 9, 9))}, Nil, Seq(correctValueAspect, wrongValueAspect))
  }
  "Test" should "pass with IO aspect and correct value" in {
    assertTesterPasses({ new IOAspectTester(5.U)} , Nil, Seq(ioAspect))
  }
  "Test" should "fail with IO aspect and incorrect value" in {
    assertTesterFails({ new IOAspectTester(6.U)} , Nil, Seq(ioAspect))
  }
}
