// SPDX-License-Identifier: Apache-2.0

package chiselTests.aop

import chisel3.testers.BasicTester
import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.aop.Select.{PredicatedConnect, When, WhenNot}
import chisel3.aop.{Aspect, Select}
import chisel3.experimental.ExtModule
import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}
import firrtl.AnnotationSeq
import chisel3.experimental.hierarchy._
import chisel3.experimental.CloneModuleAsRecord

import scala.reflect.runtime.universe.TypeTag


class SelectSpec extends ChiselFlatSpec with Utils {
  import Examples._

  "Test" should "pass if selecting correct registers" in {
    execute(
      () => new ExampleModule(Seq(0, 1, 2)),
      { dut: Definition[ExampleModule] => Select.registers(dut) },
      { dut: Definition[ExampleModule] => Seq(dut.counter) }
    )
  }

  "Test" should "pass if selecting correct wires" in {
    execute(
      () => new ExampleModule(Seq(0, 1, 2)),
      { dut: Definition[ExampleModule] => Select.wires(dut) },
      { dut: Definition[ExampleModule] => Seq(dut.values) }
    )
  }

  "Test" should "pass if selecting correct printfs" in {
    execute(
      () => new ExampleModule(Seq(0, 1, 2)),
      { dut: Definition[ExampleModule] => Seq(Select.printfs(dut).last.toString) },
      { dut: Definition[ExampleModule] =>
        Seq(Select.Printf(
          dut.p,
          Seq(
            When(Select.ops("eq")(dut).last.asInstanceOf[Bool]),
            When(dut.nreset),
            WhenNot(dut.overflow)
          ),
          dut.p.pable,
          dut.clock
        ).toString)
      }
    )
  }

  /*
  "Test" should "pass if selecting correct connections" in {
    execute(
      () => new ExampleModule(Seq(0, 1, 2)),
      { dut: ExampleModule => Select.connectionsTo(dut)(dut.counter) },
      { dut: ExampleModule =>
        Seq(PredicatedConnect(Nil, dut.counter, dut.added, false),
          PredicatedConnect(Seq(When(dut.overflow)), dut.counter, dut.zero, false))
      }
    )
  }

  "Test" should "pass if selecting ops by kind" in {
    execute(
      () => new ExampleModule(Seq(0, 1, 2)),
      { dut: ExampleModule => Select.ops("tail")(dut) },
      { dut: ExampleModule => Seq(dut.added, dut.zero) }
    )
  }

  "Test" should "pass if selecting ops" in {
    execute(
      () => new ExampleModule(Seq(0, 1, 2)),
      { dut: ExampleModule => Select.ops(dut).collect { case ("tail", d) => d} },
      { dut: ExampleModule => Seq(dut.added, dut.zero) }
    )
  }

  "Test" should "pass if selecting correct stops" in {
    execute(
      () => new ExampleModule(Seq(0, 1, 2)),
      { dut: ExampleModule => Seq(Select.stops(dut).last) },
      { dut: ExampleModule =>
        Seq(Select.Stop(
          Seq(
            When(Select.ops("eq")(dut).dropRight(1).last.asInstanceOf[Bool]),
            When(dut.nreset),
            WhenNot(dut.overflow)
          ),
          1,
          dut.clock
        ))
      }
    )
  }

  "Blackboxes" should "be supported in Select.instances" in {
    class BB extends ExtModule { }
    class Top extends RawModule {
      val bb = Module(new BB)
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
    }).elaborate
      .collectFirst { case DesignAnnotation(design: Top) => design }
      .get
    Select.collectDeep(top) { case x => x } should equal (Seq(top, top.inst0))
    Select.getDeep(top)(x => Seq(x)) should equal (Seq(top, top.inst0))
    Select.instances(top) should equal (Seq(top.inst0))
  }

  "Using Definition/Instance with Module Select" should "throw an error" in {
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
    }).elaborate
      .collectFirst { case DesignAnnotation(design: Top) => design }
      .get
    intercept[Exception] { Select.collectDeep(top) { case x => x } }
    intercept[Exception] { Select.getDeep(top)(x => Seq(x)) }
    intercept[Exception] { Select.instances(top) }
  }

  "Using Definition/Instance with Instance Select" should "work correctly" in {
    val top = ChiselGeneratorAnnotation(() => {
      new HierarchyTop()
    }).elaborate
      .collectFirst { case DesignAnnotation(design: HierarchyTop) => design }
      .get
    Select2.allInstancesOf[Child](top.toDefinition).map(_.toTarget) should equal (Seq(top.inst0.toTarget, top.inst1.toTarget))
    Select2.instancesOf[Child](top.toDefinition).map(_.toTarget) should equal (Seq(top.inst0.toTarget, top.inst1.toTarget))
    Select2.instances(top.toDefinition).map(_.toTarget) should equal (Seq(top.inst0.toTarget, top.inst1.toTarget))
  }

  "Using Definition/Instance with Instance Select" should "work correctly" in {
    val top = ChiselGeneratorAnnotation(() => {
      new HierarchyTop()
    }).elaborate
      .collectFirst { case DesignAnnotation(design: HierarchyTop) => design }
      .get
    Select2.allInstancesOf[Child](top.toDefinition).map(_.toTarget) should equal (Seq(top.inst0.toTarget, top.inst1.toTarget))
    Select2.instancesOf[Child](top.toDefinition).map(_.toTarget) should equal (Seq(top.inst0.toTarget, top.inst1.toTarget))
    Select2.instances(top.toDefinition).map(_.toTarget) should equal (Seq(top.inst0.toTarget, top.inst1.toTarget))
  }

  "Test of hierarchy" should "pass if selecting correct registers" in {
    execute(
      () => new TopTest(Seq(0, 1, 2)),
      { dut: TopTest => Select2.allInstancesOf[Select2Test](dut).flatMap(Select2.registers(_)) },
      { dut: TopTest => Seq(dut.inst0.counter, dut.inst1.counter) }
    )
  }

  "Test" should "pass if selecting correct wires" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Select.wires(dut) },
      { dut: SelectTester => Seq(dut.values) }
    )
  }

  "Test" should "pass if selecting correct printfs" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Seq(Select.printfs(dut).last.toString) },
      { dut: SelectTester =>
        Seq(Select.Printf(
          dut.p,
          Seq(
            When(Select.ops("eq")(dut).last.asInstanceOf[Bool]),
            When(dut.nreset),
            WhenNot(dut.overflow)
          ),
          dut.p.pable,
          dut.clock
        ).toString)
      }
    )
  }

  "Test" should "pass if selecting correct connections" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Select.connectionsTo(dut)(dut.counter) },
      { dut: SelectTester =>
        Seq(PredicatedConnect(Nil, dut.counter, dut.added, false),
          PredicatedConnect(Seq(When(dut.overflow)), dut.counter, dut.zero, false))
      }
    )
  }

  "Test" should "pass if selecting ops by kind" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Select.ops("tail")(dut) },
      { dut: SelectTester => Seq(dut.added, dut.zero) }
    )
  }

  "Test" should "pass if selecting ops" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Select.ops(dut).collect { case ("tail", d) => d} },
      { dut: SelectTester => Seq(dut.added, dut.zero) }
    )
  }

  "Test" should "pass if selecting correct stops" in {
    execute(
      () => new SelectTester(Seq(0, 1, 2)),
      { dut: SelectTester => Seq(Select.stops(dut).last) },
      { dut: SelectTester =>
        Seq(Select.Stop(
          Seq(
            When(Select.ops("eq")(dut).dropRight(1).last.asInstanceOf[Bool]),
            When(dut.nreset),
            WhenNot(dut.overflow)
          ),
          1,
          dut.clock
        ))
      }
    )
  }
  */

}

