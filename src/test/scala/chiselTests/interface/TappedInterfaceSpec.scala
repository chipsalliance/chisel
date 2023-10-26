// SPDX-License-Identifier: Apache-2.0
package chiselTests.interface

import chisel3._
import chisel3.probe._
import chisel3.util.experimental.BoringUtils.rwTap
import chisel3.interface.{ConformsTo, Interface}
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

/** This modifies the `separableTests.SeparableBlackBoxSpec` to make the
  * generation of that example's `InternalModuleBlackBox` and `BarWrapper` automated using
  * `separable.Interface` and `separable.ConformsTo`.
  */
class TappedInterfaceSpec extends AnyFunSpec with Matchers {

  /** This is the definition of the interface. */
  object WrapperModuleInterface extends Interface {

    /** This is the agreed-upon port-level interface. */
    final class WrapperModuleInterfaceBundle extends Bundle {
      val a = Input(Bool())
      val b = Output(Bool())
      val c = Output(RWProbe(Bool()))
      val d = Output(RWProbe(Bool()))
    }

    override type Ports = WrapperModuleInterfaceBundle

    override type Properties = Unit

    /** Generate the ports given the parameters. */
    override val ports = new Ports

  }

  object CompilationUnit1 {

    /** This is an internal part of a "DUT". Its I/O does not conform to the
      * specification-set interface.
      */
    class InternalModule extends Module {
      val x = IO(Input(Bool()))
      val y = IO(Output(Bool()))
      val lfsr = chisel3.util.random.LFSR(16)

      // things that will get tapped
      val internalWire = Wire(Bool())
      val internalReg = RegInit(Bool(), false.B)
      internalWire := lfsr(0)
      internalReg := lfsr(1)

      y := x ^ internalWire ^ internalReg
    }

    /** This wraps the above module and taps into it in order to create an
      * interface that looks like the specification-set interface.
      */
    class WrapperModule extends Module {
      val hello = IO(Input(Bool()))
      val world = IO(Output(Bool()))
      val tapWire = IO(Output(RWProbe(Bool())))
      val tapReg = IO(Output(RWProbe(Bool())))

      val internalModule = Module(new InternalModule)

      internalModule.x := hello
      world := internalModule.y

      define(tapWire, rwTap(internalModule.internalWire))
      define(tapReg, rwTap(internalModule.internalReg))
    }

    /** The owner of the "DUT" (WrapperModule) needs to write this. This defines how to
      * hook up the "DUT" to the specification-set interface.
      */
    implicit val wrapperConformance =
      new ConformsTo[WrapperModuleInterface.type, WrapperModule] {

        override def genModule() = new WrapperModule

        override def portMap = Seq(
          _.hello -> _.a,
          _.world -> _.b,
          _.tapWire -> _.c,
          _.tapReg -> _.d
        )

        override def properties = {}

      }
  }

  object CompilationUnit2 {

    /** This acts as the testbench to the DUT wrapper. This stamps out the
      * "DUT" once, but using the blackbox version of it that conforms to the
      * specification-set port list.
      *
      * This drives the probe that DUT wrapper taps.
      */
    class Foo extends Module {
      val a = IO(Input(Bool()))
      val b = IO(Output(Bool()))

      val baz = chisel3.Module(new WrapperModuleInterface.Wrapper.BlackBox)

      baz.io.a := a
      b := baz.io.b

      forceInitial(baz.io.c, true.B)
      force(baz.io.d, false.B)
    }
  }

  describe("Behavior of Interfaces") {

    it("should compile a design separably") {

      /** Now we compile the design into the "build/Interfaces" directory. Both
        * "Foo" and one copy of the "DUT", using the utility in "WrapperModuleInterface",
        * are compiled in separate processes. Finally, Verilator is run to check
        * that everything works.
        */
      val dir = new java.io.File("test_run_dir/interface/TappedInterfaceSpec/should-compile-a-design-separably")

      /** Bring the conformance into scope so that we can build the wrapper
        * module. If this is not brought into scope, trying to build a
        * `WrapperModuleInterface.Module` will fail during Scala compilation.
        */
      import CompilationUnit1.wrapperConformance

      info("compile okay!")
      Drivers.compile(
        dir,
        Drivers.CompilationUnit(() => new CompilationUnit2.Foo),
        Drivers.CompilationUnit(() => new (WrapperModuleInterface.Wrapper.Module))
      )

      info("link okay!")
      Drivers.link(dir, "compile-0/Foo.sv")

    }
  }

}
