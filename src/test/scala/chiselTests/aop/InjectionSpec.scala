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
}

class InjectionSpec extends ChiselFlatSpec {
  "Test" should "pass if inserted the correct values" in {
    assertTesterPasses{ new AspectTester(Seq(0, 1, 2)) }
  }
  "Test" should "fail if inserted the wrong values" in {
    assertTesterFails{ new AspectTester(Seq(9, 9, 9)) }
  }
  "Test" should "pass if pass wrong values, but correct with aspect" in {
    val correctValues = InjectingAspect(
      {dut: AspectTester => Seq(dut)},
      {dut: AspectTester =>
        for(i <- 0 until dut.values.length) {
          dut.values(i) := i.U
        }
      }
    )

    case object MyConcern {
      def aspects = Seq(correctValues)
    }

    assertTesterPasses({ new AspectTester(Seq(9, 9, 9))} , Nil, MyConcern.aspects)
  }
}
