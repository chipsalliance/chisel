// See LICENSE for license details.

package chiselTests.aop

import chisel3._
import chisel3.aop.Select
import chisel3.aop.injecting.{InjectingAspect, inject}
import chisel3.experimental.{ChiselAnnotation, annotate}
import chisel3.testers.BasicTester
import chiselTests.ChiselFlatSpec
import firrtl.annotations.Annotation

class ValuesModule(defaultValues: Seq[Int]) extends MultiIOModule {
  val values = VecInit(defaultValues.map(_.U))
  val out = IO(Output(Vec(values.size, UInt())))
  out := values
}

class NestedAspectTester(defaultValues: Seq[Int], correctValues: Seq[Int]) extends BasicTester {
  // Just check default and correct values are of same length
  require(defaultValues.size == correctValues.size)

  // Correct values vec
  val correctVec = VecInit(correctValues.map(_.U))

  // Create counter to check each value
  val counter = RegInit(0.U(defaultValues.length.W))
  val valueModule = Module(new ValuesModule(defaultValues))
  counter := counter + 1.U
  when(counter >= valueModule.out.length.U) {
    stop()
  }.otherwise {
    when(reset.asBool() === false.B) {
      printf("valueModule(%d) = %d, should be %d\n", counter, valueModule.out(counter), correctVec(counter))
      assert(correctVec(counter) === valueModule.out(counter))
    }
  }

  // Direct child injection
  inject(valueModule) { vm: ValuesModule =>
    for(i <- 0 until vm.values.length) {
      vm.values(i) := correctValues(i).U
    }
  }

  // Scoped programmatic child injection by type
  inject.withScope(
    this
  )(
    p => Select.collectDeep(p) { case c: ValuesModule => c }
  )
  { vm: ValuesModule =>
    for(i <- 0 until vm.values.length) {
      vm.values(i) := correctValues(i).U
    }
  }
}


class NestedInjectingSpec extends ChiselFlatSpec {

  "Test" should "pass if inserted the wrong values" in {
    assertTesterPasses{ new NestedAspectTester(Seq(0, 0, 0), Seq(1, 2, 3)) }
  }

}
