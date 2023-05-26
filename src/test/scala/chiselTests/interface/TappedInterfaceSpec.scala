// SPDX-License-Identifier: Apache-2.0
package chiselTests.interface

import chisel3._
import chisel3.probe._
import chisel3.interface.{ConformsTo, Interface}
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

/** This modifies the `separableTests.SeparableBlackBoxSpec` to make the
  * generation of that example's `BarBlackBox` and `BarWrapper` automated using
  * `separable.Interface` and `separable.ConformsTo`.
  */
class TappedInterfaceSpec extends AnyFunSpec with Matchers {

  /** This is the definition of the interface. */
  object BazInterface extends Interface {

    /** This is the agreed-upon port-level interface. */
    final class BazBundle extends Bundle {
      val a = Input(Bool())
      val b = Output(Bool())
      val c = Output(RWProbe(Bool()))
    }

    override type Ports = BazBundle

    override type Properties = Unit

    /** Generate the ports given the parameters. */
    override val ports = new Ports

  }

  object CompilationUnit1 {

    /** This is an internal part of a "DUT". Its I/O does not conform to the
      * specification-set interface.
      */
    class Bar extends RawModule {
      val x = IO(Input(Bool()))
      val y = IO(Output(Bool()))
      val lfsr = chisel3.util.random.LFSR(1)
      val z = Wire(Bool())
      z := lfsr
      dontTouch(z)
      y := x ^ z
    }

    /** This wraps the above module and taps into it in order to create an
      * interface that looks like the specification-set interface.
      */
    class Baz extends RawModule {
      val hello = IO(Input(Bool()))
      val world = IO(Output(Bool()))
      val goodbye = IO(Output(RWProbe(Bool())))

      val bar = Module(new Bar)

      bar.x := hello
      world := bar.y

      goodbye := chisel3.util.experimental.BoringUtils.tap(bar.z)
    }

    /** The owner of the "DUT" (Bar) needs to write this. This defines how to
      * hook up the "DUT" to the specification-set interface.
      */
    implicit val bazConformance =
      new ConformsTo[BazInterface.type, Baz] {

        override def genModule() = new Baz

        override def portMap = Seq(
          _.hello -> _.a,
          _.world -> _.b,
          _.goodbye -> _.c
        )

        override def properties = {}

      }
  }

  object CompilationUnit2 {

    /** This acts as the testbench to the "DUT" (Baz). This stamps out the
      * "DUT" once, but using the blackbox version of it that conforms to the
      * specification-set port list.
      *
      * This drives the probe that Baz tapped from Bar.
      */
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      val b = IO(Output(Bool()))

      val baz = chisel3.Module(new BazInterface.Wrapper.BlackBox)

      baz.io.a := a
      b := baz.io.b

      forceInitial(baz.io.c, true.B)
    }
  }

  describe("Behavior of Interfaces") {

    it("should compile a design separably") {

      /** Now we compile the design into the "build/Interfaces" directory. Both
        * "Foo" and one copy of the "DUT", using the utility in "BazInterface",
        * are compiled in separate processes. Finally, Verilator is run to check
        * that everything works.
        */
      val dir = new java.io.File("test_run_dir/interface/TappedInterfaceSpec/should-compile-a-design-separably")

      /** Bring the conformance into scope so that we can build the wrapper
        * module. If this is not brought into scope, trying to build a
        * `BazInterface.Module` will fail during Scala compilation.
        */
      import CompilationUnit1.bazConformance

      println(circt.stage.ChiselStage.emitCHIRRTL(new CompilationUnit2.Foo))

      info("compile okay!")
      Drivers.compile(
        dir,
        Drivers.CompilationUnit(() => new CompilationUnit2.Foo),
        Drivers.CompilationUnit(() => new (BazInterface.Wrapper.Module))
      )

      info("link okay!")
      Drivers.link(dir, "compile-0/Foo.sv")

    }
  }

  describe("Basic behavior of properties") {

    // A container of the types of properties that are on this Interface.
    case class SomeProperties(a: Int, b: String, c: Int => Int)

    object InterfaceWithProperties extends Interface {
      override type Ports = Bundle {}
      override type Properties = SomeProperties
      override val ports = new Bundle {}
    }

    class Bar extends RawModule

    val conformanceGood = new ConformsTo[InterfaceWithProperties.type, Bar] {
      override def genModule() = new Bar
      override def portMap = Seq.empty
      // The conformance defines the properties.
      override def properties = SomeProperties(42, "hello", _ + 8)
    }

    val conformanceBad = new ConformsTo[InterfaceWithProperties.type, Bar] {
      override def genModule() = new Bar
      override def portMap = Seq.empty
      // The conformance defines the properties.
      override def properties = SomeProperties(51, "hello", _ + 8)
    }

    it("should be able to read properties") {
      class Foo(conformance: ConformsTo[InterfaceWithProperties.type, Bar]) extends RawModule {
        private implicit val c = conformance

        val bar = Module(new InterfaceWithProperties.Wrapper.BlackBox)
        private val properties = bar.properties[Bar]

        // Check that the component works in this context.
        require(properties.a <= 50)
      }

      info("a conformance with suitable properties should work")
      ChiselStage.emitCHIRRTL(new Foo(conformanceGood))

      info("a conformance with unsuitable properties should error")
      an[IllegalArgumentException] shouldBe thrownBy(ChiselStage.emitCHIRRTL(new Foo(conformanceBad)))
    }

  }

  describe("Error behavior of Interfaces") {

    it("should error if an Interface is not connected to") {

      class Qux extends RawModule {
        val a = IO(Input(Bool()))
      }

      implicit val quxConformance = new ConformsTo[BazInterface.type, Qux] {

        override def genModule() = new Qux

        override def portMap = Seq()

        override def properties = ()
      }

      val exception = the[Exception] thrownBy circt.stage.ChiselStage.emitCHIRRTL(new (BazInterface.Wrapper.Module))

      exception.getMessage() should include("unable to conform module 'Qux' to interface 'BazInterface'")
    }

  }

}
