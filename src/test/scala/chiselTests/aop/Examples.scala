package chiselTests.aop

import chisel3._
import chisel3.experimental.hierarchy._

object Examples {
  class HierarchyTop extends Module {
    val in = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
    val definition = Definition(new Child)
    val inst0 = Instance(definition)
    val inst1 = Instance(definition)
    inst0.in := in
    inst1.in := inst0.out
    out := inst1.out
  }

  @instantiable
  class Child extends RawModule {
    @public val in = IO(Input(UInt(8.W)))
    @public val out = IO(Output(UInt(8.W)))
    out := in
  }

  class TwoExamples(results: Seq[Int]) extends Module {
    val definition = Definition(new ExampleModule(results))
    val inst0 = Instance(definition)
    val inst1 = Instance(definition)
  }

  @instantiable
  class ExampleModule(results: Seq[Int]) extends Module {
    @public val values = VecInit(results.map(_.U))
    @public val counter = RegInit(0.U(results.length.W))
    @public val added = counter + 1.U
    counter := added
    @public val overflow = counter >= values.length.U
    @public val nreset = reset.asBool() === false.B
    @public val selected = values(counter)
    @public val zero = 0.U + 0.U
    var p: printf.Printf = null
    when(overflow) {
      counter := zero
      stop()
    }.otherwise {
      when(nreset) {
        assert(counter === values(counter))
        p = printf("values(%d) = %d\n", counter, selected)
      }
  
    }
  }

}