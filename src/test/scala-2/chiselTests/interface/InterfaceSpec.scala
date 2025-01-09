// SPDX-License-Identifier: Apache-2.0
package chiselTests.interface

import chisel3._
import chisel3.interface.{ConformsTo, Interface}
import chisel3.probe._
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec
import org.scalatest.matchers.should.Matchers

/** This modifies the `separableTests.SeparableBlackBoxSpec` to make the
  * generation of that example's `BarBlackBox` and `BarWrapper` automated using
  * `separable.Interface` and `separable.ConformsTo`.
  */
class InterfaceSpec extends AnyFunSpec with Matchers {

  /** This is the definition of the interface. */
  object BarInterface extends Interface {

    /** This is the agreed-upon port-level interface. */
    final class BarBundle extends Bundle {
      val a = Input(Bool())
      val b = Output(Bool())
    }

    override type Ports = BarBundle

    override type Properties = Unit

    /** Generate the ports given the parameters. */
    override val ports = new Ports

  }

  object CompilationUnit1 {

    /** This is the "DUT" that will be compiled in one compilation unit and
      * reused multiple times. The designer is free to do whatever they want
      * with this, internally or at the boundary. The port-level interface of
      * this module does not align with the interface.
      */
    class Bar extends RawModule {
      val x = IO(Input(Bool()))
      val y = IO(Output(Bool()))
      y := ~x
    }

    /** The owner of the "DUT" (Bar) needs to write this. This defines how to
      * hook up the "DUT" to the specification-set interface.
      */
    implicit val barConformance =
      new ConformsTo[BarInterface.type, Bar] {

        override def genModule() = new Bar

        override def portMap = Seq(
          _.x -> _.a,
          _.y -> _.b
        )

        override def properties = {}

      }
  }

  object CompilationUnit2 {

    /** This is an alternative "DUT" that is differnet from the actual DUT, Bar.
      * It differs in its ports and internals.
      */
    class Baz extends RawModule {
      val hello = IO(Input(Bool()))
      val world = IO(Output(Bool()))
      world := hello
    }

    /** The owner of the "DUT" (Bar) needs to write this. This defines how to
      * hook up the "DUT" to the specification-set interface.
      */
    implicit val bazConformance =
      new ConformsTo[BarInterface.type, Baz] {

        override def genModule() = new Baz

        override def portMap = Seq(
          _.hello -> _.a,
          _.world -> _.b
        )

        override def properties = {}

      }
  }

  object CompilationUnit3 {

    /** This is a module above the "DUT" (Bar). This stamps out the "DUT" twice,
      * but using the blackbox version of it that conforms to the
      * specification-set port list.
      */
    class Foo extends RawModule {
      val a = IO(Input(Bool()))
      val b = IO(Output(Bool()))

      val bar1, bar2 = chisel3.Module(new BarInterface.Wrapper.BlackBox)

      bar1.io.a := a
      bar2.io.a := bar1.io.b
      b := bar2.io.b
    }
  }

  describe("Behavior of Interfaces") {

    it("should compile a design separably") {

      /** Now we compile the design into the "build/Interfaces" directory. Both
        * "Foo" and one copy of the "DUT", using the utility in "BarInterface",
        * are compiled in separate processes. Finally, Verilator is run to check
        * that everything works.
        */
      val dir = new java.io.File("test_run_dir/interface/InterfaceSpec/should-compile-a-design-separably")

      /** Bring the conformance into scope so that we can build the wrapper
        * module. If this is not brought into scope, trying to build a
        * `BarInterface.Module` will fail during Scala compilation.
        */
      import CompilationUnit1.barConformance

      info("compile okay!")
      Drivers.compile(
        dir,
        Drivers.CompilationUnit(() => new CompilationUnit3.Foo),
        Drivers.CompilationUnit(() => new (BarInterface.Wrapper.Module))
      )

      info("link okay!")
      Drivers.link(dir, "compile-0/Foo.sv")

    }

    it("should compile an alternative design separately") {

      val dir = new java.io.File("test_run_dir/interface/InterfaceSpec/should-compile-an-alternative-design-separably")

      import CompilationUnit2.bazConformance

      info("compile okay!")
      Drivers.compile(
        dir,
        Drivers.CompilationUnit(() => new CompilationUnit3.Foo),
        Drivers.CompilationUnit(() => new (BarInterface.Wrapper.Module))
      )

      info("link okay!")
      Drivers.link(dir, "compile-0/Foo.sv")

    }

    it("should compile a stub") {

      val dir = new java.io.File("test_run_dir/interface/InterfaceSpec/should-compile-a-stub")

      import CompilationUnit2.bazConformance

      circt.stage.ChiselStage.emitCHIRRTL(new (BarInterface.Wrapper.Stub)(()), Array("--full-stacktrace"))

      info("compile okay!")
      Drivers.compile(
        dir,
        Drivers.CompilationUnit(() => new CompilationUnit3.Foo),
        Drivers.CompilationUnit(() => new (BarInterface.Wrapper.Stub)(()))
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

      implicit val quxConformance = new ConformsTo[BarInterface.type, Qux] {

        override def genModule() = new Qux

        override def portMap = Seq()

        override def properties = ()
      }

      val exception = the[Exception] thrownBy circt.stage.ChiselStage.emitCHIRRTL(new (BarInterface.Wrapper.Module))

      exception.getMessage() should include("unable to conform module 'Qux' to interface 'BarInterface'")
    }

  }

  describe("Ref types in Interfaces") {

    it("should support ref types") {
      object RefInterface extends Interface {

        final class RefBundle extends Bundle {
          val r = Output(Probe(Bool()))
        }

        override type Ports = RefBundle

        override type Properties = Unit

        override val ports = new Ports

      }

      class RefComponent extends RawModule {
        val w_ref = IO(Output(Probe(Bool())))
        val w = WireInit(false.B)
        val w_probe = ProbeValue(w)
        define(w_ref, w_probe)
      }

      implicit val refConformance =
        new ConformsTo[RefInterface.type, RefComponent] {
          override def genModule() = new RefComponent

          override def portMap = Seq(
            _.w_ref -> _.r
          )

          override def properties = {}
        }

      class RefClient extends RawModule {
        val x = IO(Output(Bool()))
        val refInterface = chisel3.Module(new RefInterface.Wrapper.BlackBox)
        x := read(refInterface.io.r)
      }

      val dir = new java.io.File("test_run_dir/interface/InterfaceSpec/should-support-ref-types")
      Drivers.compile(
        dir,
        Drivers.CompilationUnit(() => new RefClient),
        Drivers.CompilationUnit(() => new RefInterface.Wrapper.Module)
      )
      Drivers.link(dir, "compile-0/RefClient.sv")
    }

  }

  it("should support ref types that point to subfields of aggregates") {
    object RefInterface extends Interface {

      final class RefBundle extends Bundle {
        val r = Output(Probe(Bool()))
      }

      override type Ports = RefBundle

      override type Properties = Unit

      override val ports = new Ports

    }

    class RefComponent extends RawModule {
      val w_ref = IO(Output(Probe(Bool())))
      val w = Wire(new Bundle {
        val a = UInt(4.W)
        val b = Bool()
      })
      w.a := 2.U(4.W)
      w.b := true.B
      val w_probe = ProbeValue(w.b)
      define(w_ref, w_probe)
    }

    implicit val refConformance =
      new ConformsTo[RefInterface.type, RefComponent] {
        override def genModule() = new RefComponent

        override def portMap = Seq(
          _.w_ref -> _.r
        )

        override def properties = {}
      }

    class RefClient extends RawModule {
      val x = IO(Output(Bool()))
      val refInterface = chisel3.Module(new RefInterface.Wrapper.BlackBox)
      x := read(refInterface.io.r)
    }

    val dir = new java.io.File("test_run_dir/interface/InterfaceSpec/should-support-ref-types-to-aggregates")
    Drivers.compile(
      dir,
      Drivers.CompilationUnit(() => new RefClient),
      Drivers.CompilationUnit(() => new RefInterface.Wrapper.Module)
    )
    Drivers.link(dir, "compile-0/RefClient.sv")
  }

}
