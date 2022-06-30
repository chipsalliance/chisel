// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chiselTests.ChiselFlatSpec

class LazyCloneSpec extends ChiselFlatSpec {
  behavior.of("LazyClone")

  it should "lazily clone" in {
    object Foo {
      var count = 0L
    }

    class Foo extends Bundle {
      val a = UInt(8.W)
      Foo.count += 1
    }

    class MyModule extends RawModule {
      val io = IO(Flipped(new Bundle {
        val foo = Output(new Foo)
        val bar = Input(new Foo)
      }))
    }
    ChiselStage.elaborate(new MyModule)
    println(s"copies: ${Foo.count}")
    Foo.count should be < 12L
  }
}
