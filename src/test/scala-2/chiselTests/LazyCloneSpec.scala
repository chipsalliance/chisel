// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import collection.immutable.VectorMap

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

  class GenRecord(gen: UInt) extends Record {
    lazy val elements = VectorMap("foo" -> gen)
    Counter.count += 1
  }

  class NestedGenBundle(gen: UInt) extends Bundle {
    val foo = new Bundle {
      val bar = gen
    }
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
    ChiselStage.emitCHIRRTL(new MyModule)
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
    ChiselStage.emitCHIRRTL(new MyModule)
    Counter.count should be(3L)
  }

  it should "not clone when ref is external to the Bundle but not the binding operation" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val io = IO(Flipped(new Bundle {
        val foo = Output(new Bar(UInt(8.W)))
        val bar = Input(new Bar(UInt(8.W)))
      }))
    }
    ChiselStage.emitCHIRRTL(new MyModule)
    Counter.count should be(2L)
  }

  it should "clone Records because of external refs" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val gen = UInt(8.W)
      val in = IO(Input(new GenRecord(gen)))
      val out = IO(Output(new GenRecord(gen)))
      out := in
    }
    ChiselStage.emitCHIRRTL(new MyModule)
    Counter.count should be(4L)
  }

  it should "not clone when ref is external to the Record but not the binding operation" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val in = IO(Input(new GenRecord(UInt(8.W))))
      val out = IO(Output(new GenRecord(UInt(8.W))))
      out := in
    }
    ChiselStage.emitCHIRRTL(new MyModule)
    Counter.count should be(2L)
  }

  it should "clone because of nested external ref" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val gen = UInt(8.W)
      val in = IO(Input(new NestedGenBundle(gen)))
      val out = IO(Output(new NestedGenBundle(gen)))
      out := in
    }
    ChiselStage.emitCHIRRTL(new MyModule)
    Counter.count should be(4L)
  }

  it should "not clone when nested ref is external to the Bundle but not the binding operation" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val in = IO(Input(new NestedGenBundle(UInt(8.W))))
      val out = IO(Output(new NestedGenBundle(UInt(8.W))))
      out := in
    }
    ChiselStage.emitCHIRRTL(new MyModule)
    Counter.count should be(2L)
  }

  it should "not clone Vecs (but Vecs always clone their gen 1 + size times)" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val in = IO(Input(Vec(2, new Foo)))
      val out = IO(Output(Vec(2, new Foo)))
      out := in
    }
    ChiselStage.emitCHIRRTL(new MyModule)
    // Each Vec has 3 clones of Foo + the original Foo, then the Vec isn't cloned
    Counter.count should be(8L)
  }

  it should "not clone Vecs even with external refs (but Vecs always clone their gen 1 + size times)" in {
    Counter.count = 0L
    class MyModule extends RawModule {
      val gen = new Foo
      val in = IO(Input(Vec(2, gen)))
      val out = IO(Output(Vec(2, gen)))
      out := in
    }
    ChiselStage.emitCHIRRTL(new MyModule)
    // Each Vec has 3 clones of Foo + the original Foo, then the Vec isn't cloned
    Counter.count should be(7L)
  }
}
