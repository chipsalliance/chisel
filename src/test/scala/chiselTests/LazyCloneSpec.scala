// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chiselTests.ChiselFlatSpec

class LazyCloneSpec extends ChiselFlatSpec {
  object Counter {
    var count = 0L
  }

  class Foo extends Bundle {
    val a = UInt(8.W)
    Counter.count += 1
  }

  class Bar(x: UInt) extends Bundle {
    val a = x
    Counter.count += 1
  }

  behavior.of("LazyClone")

  it should "not clone" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val io = IO(Flipped(new Bundle {
        val x = Output(new Foo)
        val y = Input(new Foo)
      }))
    }
    ChiselStage.elaborate(new MyModule)
    Counter.count should be(2L)
  }

  it should "share with cloning" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val foo = new Foo
      val io = IO(Flipped(new Bundle {
        val x = Output(foo)
        val y = Input(foo)
      }))
    }
    ChiselStage.elaborate(new MyModule)
    Counter.count should be(3L)
  }

  it should "clone because of external ref" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val io = IO(Flipped(new Bundle {
        val foo = Output(new Bar(UInt(8.W)))
        val bar = Input(new Bar(UInt(8.W)))
      }))
    }
    ChiselStage.elaborate(new MyModule)
    Counter.count should be(4L)
  }
}
