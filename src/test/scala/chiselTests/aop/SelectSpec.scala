// SPDX-License-Identifier: Apache-2.0

package chiselTests.aop

import chisel3.testers.BasicTester
import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.aop.Select.{PredicatedConnect, When, WhenNot}
import chisel3.aop.Select
import chisel3.experimental.ExtModule
import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}
import circt.stage.ChiselStage
import firrtl.AnnotationSeq

import scala.reflect.runtime.universe.TypeTag

class SelectTester(results: Seq[Int]) extends BasicTester {
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

class SelectSpec extends ChiselFlatSpec {

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
    }).elaborate.collectFirst { case DesignAnnotation(design: Top) => design }.get
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
    }).elaborate.collectFirst { case DesignAnnotation(design: Top) => design }.get
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
}
