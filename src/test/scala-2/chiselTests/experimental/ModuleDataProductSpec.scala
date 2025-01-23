// See LICENSE for license details.

package chiselTests.experimental

import chisel3._
import chisel3.experimental.{BaseModule, ExtModule}
import chisel3.experimental.dataview.DataProduct
import chiselTests.ChiselFlatSpec

object ModuleDataProductSpec {
  class MyBundle extends Bundle {
    val foo = UInt(8.W)
    val bar = Vec(1, UInt(8.W))
  }
  trait MyIntf extends BaseModule {
    val in = IO(Input(new MyBundle))
    val out = IO(Output(new MyBundle))
  }
  class Passthrough extends RawModule {
    val in = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
    out := in
  }
  class MyUserModule extends Module with MyIntf {
    val inst = Module(new Passthrough)
    inst.in := in.foo
    val r = RegNext(in)
    out := r
  }

  class MyExtModule extends ExtModule with MyIntf
  class MyExtModuleWrapper extends RawModule with MyIntf {
    val inst = Module(new MyExtModule)
    inst.in := in
    out := inst.out
  }
}

class ModuleDataProductSpec extends ChiselFlatSpec {
  import ModuleDataProductSpec._

  behavior.of("DataProduct")

  it should "work for UserModules (recursively)" in {
    val m = elaborateAndGetModule(new MyUserModule)
    val expected = Seq(
      m.clock -> "m.clock",
      m.reset -> "m.reset",
      m.in -> "m.in",
      m.in.foo -> "m.in.foo",
      m.in.bar -> "m.in.bar",
      m.in.bar(0) -> "m.in.bar(0)",
      m.out -> "m.out",
      m.out.foo -> "m.out.foo",
      m.out.bar -> "m.out.bar",
      m.out.bar(0) -> "m.out.bar(0)",
      m.r -> "m.r",
      m.r.foo -> "m.r.foo",
      m.r.bar -> "m.r.bar",
      m.r.bar(0) -> "m.r.bar(0)",
      m.inst.in -> "m.inst.in",
      m.inst.out -> "m.inst.out"
    )

    val impl = implicitly[DataProduct[MyUserModule]]
    val set = impl.dataSet(m)
    for ((d, _) <- expected) {
      set(d) should be(true)
    }
    val it = impl.dataIterator(m, "m")
    it.toList should contain theSameElementsAs (expected)
  }

  it should "work for (wrapped) ExtModules" in {
    val m = elaborateAndGetModule(new MyExtModuleWrapper).inst
    val expected = Seq(
      m.in -> "m.in",
      m.in.bar -> "m.in.bar",
      m.in.bar(0) -> "m.in.bar(0)",
      m.in.foo -> "m.in.foo",
      m.out -> "m.out",
      m.out.bar -> "m.out.bar",
      m.out.bar(0) -> "m.out.bar(0)",
      m.out.foo -> "m.out.foo"
    )

    val impl = implicitly[DataProduct[MyExtModule]]
    val set = impl.dataSet(m)
    for ((d, _) <- expected) {
      set(d) should be(true)
    }
    val it = impl.dataIterator(m, "m")
    it.toList should contain theSameElementsAs (expected)
  }

}
