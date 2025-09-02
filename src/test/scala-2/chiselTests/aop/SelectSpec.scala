// SPDX-License-Identifier: Apache-2.0

package chiselTests.aop

import chisel3._
import chisel3.aop.Select
import chisel3.aop.Select.{PredicatedConnect, When, WhenNot}
import chisel3.experimental.ExtModule
import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}
import circt.stage.ChiselStage
import firrtl.AnnotationSeq
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import scala.reflect.runtime.universe.TypeTag

class SelectTester(results: Seq[Int]) extends Module {
  val values = VecInit(results.map(_.U))
  val counter = RegInit(0.U(results.length.W))
  val added = counter + 1.U
  counter := added
  val overflow = counter >= values.length.U
  val nreset = reset.asBool === false.B
  val selected = values(counter)
  val zero = 0.U + 0.U
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

class SelectSpec extends AnyFlatSpec with Matchers {

  "Test" should "pass if selecting correct registers" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.registers(dut) should be(Seq(dut.counter))
  }

  "Test" should "pass if selecting correct wires" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.wires(dut) should be(Seq(dut.values))

  }

  "Test" should "pass if selecting correct printfs" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Seq(Select.printfs(dut).last.toString) should be(
      Seq(
        Select
          .Printf(
            dut.p,
            Seq(
              When(Select.ops("eq")(dut).last.asInstanceOf[Bool]),
              When(dut.nreset),
              WhenNot(dut.overflow)
            ),
            dut.p.pable,
            dut.clock
          )
          .toString
      )
    )
  }

  "Test" should "pass if selecting correct connections" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.connectionsTo(dut)(dut.counter) should be(
      Seq(
        PredicatedConnect(Nil, dut.counter, dut.added, false),
        PredicatedConnect(Seq(When(dut.overflow)), dut.counter, dut.zero, false)
      )
    )

  }

  "Test" should "pass if selecting attach" in {
    import chisel3.experimental.{attach, Analog}
    class AttachTest extends RawModule {
      val a, b, c = IO(Analog(8.W))
      attach(a, b, c)
    }
    val dut = ChiselGeneratorAnnotation(() => new AttachTest)
      .elaborate(1)
      .asInstanceOf[DesignAnnotation[AttachTest]]
      .design
    Select.attachedTo(dut)(dut.a) should be(Set(dut.a, dut.b, dut.c))

  }

  "Test" should "pass if selecting ops by kind" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.ops("tail")(dut) should be(Seq(dut.added, dut.zero))
  }

  "Test" should "pass if selecting ops" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.ops(dut).collect { case ("tail", d) => d } should be(Seq(dut.added, dut.zero))
  }

  "Test" should "pass if selecting correct stops" in {
    val dut = ChiselGeneratorAnnotation(() => {
      new SelectTester(Seq(0, 1, 2))
    }).elaborate(1).asInstanceOf[DesignAnnotation[SelectTester]].design
    Select.stops(dut) should be(
      Seq(
        Select.Stop(
          Seq(
            When(Select.ops("eq")(dut)(1).asInstanceOf[Bool]),
            When(dut.overflow)
          ),
          0,
          dut.clock
        )
      )
    )
  }

  "Blackboxes" should "be supported in Select.instances" in {
    class BB extends ExtModule {}
    class Top extends RawModule {
      val bb = Module(new BB)
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate(1).asInstanceOf[DesignAnnotation[Top]].design
    val bbs = Select.collectDeep(top) { case b: BB => b }
    assert(bbs.size == 1)
  }

  "collectDeep" should "should look in when regions" in {
    class BB extends ExtModule {}
    class Top extends RawModule {
      when(true.B) {
        val bb = Module(new BB)
      }
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate(1).asInstanceOf[DesignAnnotation[Top]].design
    val bbs = Select.collectDeep(top) { case b: BB => b }
    assert(bbs.size == 1)
  }

  "collectDeep" should "should look in layer regions" in {
    object TestLayer extends layer.Layer(layer.LayerConfig.Extract())
    class BB extends ExtModule {}
    class Top extends RawModule {
      layer.block(TestLayer) {
        val bb = Module(new BB)
      }
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate(1).asInstanceOf[DesignAnnotation[Top]].design
    val bbs = Select.collectDeep(top) { case b: BB => b }
    assert(bbs.size == 1)
  }

  "CloneModuleAsRecord" should "NOT show up in Select aspects" in {
    import chisel3.experimental.CloneModuleAsRecord
    class Child extends RawModule {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      out := in
    }
    class Top extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      val inst0 = Module(new Child)
      val inst1 = CloneModuleAsRecord(inst0)
      inst0.in := in
      inst1("in") := inst0.out
      out := inst1("out")
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate.collectFirst { case DesignAnnotation(design: Top, _) => design }.get
    Select.collectDeep(top) { case x => x } should equal(Seq(top, top.inst0))
    Select.getDeep(top)(x => Seq(x)) should equal(Seq(top, top.inst0))
    Select.instances(top) should equal(Seq(top.inst0))
  }

  "Using Definition/Instance with Select APIs" should "throw an error" in {
    import chisel3.experimental.CloneModuleAsRecord
    import chisel3.experimental.hierarchy._
    @instantiable
    class Child extends RawModule {
      @public val in = IO(Input(UInt(8.W)))
      @public val out = IO(Output(UInt(8.W)))
      out := in
    }
    class Top extends Module {
      val in = IO(Input(UInt(8.W)))
      val out = IO(Output(UInt(8.W)))
      val definition = Definition(new Child)
      val inst0 = Instance(definition)
      val inst1 = Instance(definition)
      inst0.in := in
      inst1.in := inst0.out
      out := inst1.out
    }
    val top = ChiselGeneratorAnnotation(() => {
      new Top()
    }).elaborate.collectFirst { case DesignAnnotation(design: Top, _) => design }.get
    intercept[Exception] { Select.collectDeep(top) { case x => x } }
    intercept[Exception] { Select.getDeep(top)(x => Seq(x)) }
    intercept[Exception] { Select.instances(top) }
  }

  "Select currentInstancesIn and allCurrentInstancesIn" should "support module, extmodule, and D/I" in {
    import chisel3.experimental.hierarchy._

    class MyLeafExtModule extends ExtModule {}

    class MyLeafModule extends RawModule {}

    class MyIntermediateModule extends RawModule {
      Module(new MyLeafModule)
      Module(new MyLeafModule)
    }

    @instantiable
    class MyLeafInstance extends RawModule {}

    @instantiable
    class MyIntermediateInstance extends RawModule {
      val definition = Definition(new MyLeafInstance)
      Instance(definition)
      Instance(definition)
    }

    class MyWrapperModule extends RawModule {
      implicit val mg = new chisel3.internal.MacroGenerated {}

      val definition = Definition(new MyIntermediateInstance)

      Module(new MyIntermediateModule)

      Instance(definition)

      // Check instances thus far.
      val myInstances0 = Select.unsafe.currentInstancesIn(this).map(_._lookup { m => m.name })

      myInstances0 should be(Seq("MyIntermediateModule", "MyIntermediateInstance"))

      Module(new MyLeafExtModule)

      // Check that the new instance is also returned.
      val myInstances1 = Select.unsafe.currentInstancesIn(this).map(_._lookup { m => m.name })

      myInstances1 should be(Seq("MyIntermediateModule", "MyIntermediateInstance", "MyLeafExtModule"))
    }

    class MyTopModule extends RawModule {
      implicit val mg = new chisel3.internal.MacroGenerated {}

      Module(new MyWrapperModule)

      // Check the recursive version goes all the way down.
      val allInstances = Select.unsafe.allCurrentInstancesIn(this).map(_._lookup { m => m.name })

      allInstances should be(
        Seq(
          "MyWrapperModule",
          "MyIntermediateModule",
          "MyIntermediateInstance",
          "MyLeafExtModule",
          "MyLeafModule",
          "MyLeafModule_1",
          "MyLeafInstance",
          "MyLeafInstance"
        )
      )
    }

    ChiselStage.emitCHIRRTL(new MyTopModule)
  }

  "Looking up instances" should "not affect the result of currentInstancesIn" in {
    import chisel3.experimental.hierarchy._

    @instantiable
    class Grandchild extends Module

    @instantiable
    class Child extends Module {
      @public val g1 = Instantiate(new Grandchild)
    }

    class Parent extends Module {
      val c1 = Instantiate(new Child)

      Select.unsafe.currentInstancesIn(this) should be(Seq(c1))

      c1.g1 // This does more than you think :)

      Select.unsafe.currentInstancesIn(this) should be(Seq(c1))

      // We can't check the exact contents because what this returns is not the same object as c1.g1
      // TODO is it a problem that it's not the same object?
      Select.unsafe.allCurrentInstancesIn(this).size should be(2)

      Select.unsafe.currentInstancesIn(this) should be(Seq(c1))
    }

    ChiselStage.elaborate(new Parent)
  }
}
