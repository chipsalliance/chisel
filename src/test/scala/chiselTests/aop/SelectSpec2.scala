//// SPDX-License-Identifier: Apache-2.0
//
//package chiselTests.aop
//
//import chisel3.testers.BasicTester
//import chiselTests.ChiselFlatSpec
//import chisel3._
//import chisel3.aop.Select.{PredicatedConnect, When, WhenNot}
//import chisel3.aop.{Aspect, Select}
//import chisel3.experimental.ExtModule
//import chisel3.experimental.hierarchy._
//import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}
//import firrtl.AnnotationSeq
//
//import scala.reflect.runtime.universe.TypeTag
//
//
//class SelectSpec2 extends ChiselFlatSpec {
//
//  def execute[T <: RawModule, X](dut: () => T, selector: T => Seq[X], desired: T => Seq[X])(implicit tTag: TypeTag[T]): Unit = {
//    val ret = new chisel3.stage.ChiselStage().run(
//      Seq(
//        new chisel3.stage.ChiselGeneratorAnnotation(dut),
//        SelectAspect(selector, desired),
//        new chisel3.stage.ChiselOutputFileAnnotation("test_run_dir/Select.fir")
//      )
//    )
//  }
//
//  "Test" should "pass if selecting correct registers" in {
//    execute(
//      () => new SelectTester(Seq(0, 1, 2)),
//      { dut: SelectTester => Select.registers(dut) },
//      { dut: SelectTester => Seq(dut.counter) }
//    )
//  }
//
//  "Test" should "pass if selecting correct wires" in {
//    execute(
//      () => new SelectTester(Seq(0, 1, 2)),
//      { dut: SelectTester => Select.wires(dut) },
//      { dut: SelectTester => Seq(dut.values) }
//    )
//  }
//
//  "Test" should "pass if selecting correct printfs" in {
//    execute(
//      () => new SelectTester(Seq(0, 1, 2)),
//      { dut: SelectTester => Seq(Select.printfs(dut).last.toString) },
//      { dut: SelectTester =>
//        Seq(Select.Printf(
//          dut.p,
//          Seq(
//            When(Select.ops("eq")(dut).last.asInstanceOf[Bool]),
//            When(dut.nreset),
//            WhenNot(dut.overflow)
//          ),
//          dut.p.pable,
//          dut.clock
//        ).toString)
//      }
//    )
//  }
//
//  "Test" should "pass if selecting correct connections" in {
//    execute(
//      () => new SelectTester(Seq(0, 1, 2)),
//      { dut: SelectTester => Select.connectionsTo(dut)(dut.counter) },
//      { dut: SelectTester =>
//        Seq(PredicatedConnect(Nil, dut.counter, dut.added, false),
//          PredicatedConnect(Seq(When(dut.overflow)), dut.counter, dut.zero, false))
//      }
//    )
//  }
//
//  "Test" should "pass if selecting ops by kind" in {
//    execute(
//      () => new SelectTester(Seq(0, 1, 2)),
//      { dut: SelectTester => Select.ops("tail")(dut) },
//      { dut: SelectTester => Seq(dut.added, dut.zero) }
//    )
//  }
//
//  "Test" should "pass if selecting ops" in {
//    execute(
//      () => new SelectTester(Seq(0, 1, 2)),
//      { dut: SelectTester => Select.ops(dut).collect { case ("tail", d) => d} },
//      { dut: SelectTester => Seq(dut.added, dut.zero) }
//    )
//  }
//
//  "Test" should "pass if selecting correct stops" in {
//    execute(
//      () => new SelectTester(Seq(0, 1, 2)),
//      { dut: SelectTester => Seq(Select.stops(dut).last) },
//      { dut: SelectTester =>
//        Seq(Select.Stop(
//          Seq(
//            When(Select.ops("eq")(dut).dropRight(1).last.asInstanceOf[Bool]),
//            When(dut.nreset),
//            WhenNot(dut.overflow)
//          ),
//          1,
//          dut.clock
//        ))
//      }
//    )
//  }
//
//  "Blackboxes" should "be supported in Select.instances" in {
//    class BB extends ExtModule { }
//    class Top extends RawModule {
//      val bb = Module(new BB)
//    }
//    val top = ChiselGeneratorAnnotation(() => {
//      new Top()
//    }).elaborate(1).asInstanceOf[DesignAnnotation[Top]].design
//    val bbs = Select.collectDeep(top) { case b: BB => b }
//    assert(bbs.size == 1)
//  }
//
//  "CloneModuleAsRecord" should "NOT show up in Select aspects" in {
//    import chisel3.experimental.CloneModuleAsRecord
//    class Child extends RawModule {
//      val in = IO(Input(UInt(8.W)))
//      val out = IO(Output(UInt(8.W)))
//      out := in
//    }
//    class Top extends Module {
//      val in = IO(Input(UInt(8.W)))
//      val out = IO(Output(UInt(8.W)))
//      val inst0 = Module(new Child)
//      val inst1 = CloneModuleAsRecord(inst0)
//      inst0.in := in
//      inst1("in") := inst0.out
//      out := inst1("out")
//    }
//    val top = ChiselGeneratorAnnotation(() => {
//      new Top()
//    }).elaborate
//      .collectFirst { case DesignAnnotation(design: Top) => design }
//      .get
//    Select.collectDeep(top) { case x => x } should equal (Seq(top, top.inst0))
//    Select.getDeep(top)(x => Seq(x)) should equal (Seq(top, top.inst0))
//    Select.instances(top) should equal (Seq(top.inst0))
//  }
//
//  "Using Definition/Instance with Injecting Aspects" should "throw an error" in {
//    import chisel3.experimental.CloneModuleAsRecord
//    import chisel3.experimental.hierarchy._
//    @instantiable
//    class Child extends RawModule {
//      @public val in = IO(Input(UInt(8.W)))
//      @public val out = IO(Output(UInt(8.W)))
//      out := in
//    }
//    class Top extends Module {
//      val in = IO(Input(UInt(8.W)))
//      val out = IO(Output(UInt(8.W)))
//      val definition = Definition(new Child)
//      val inst0 = Instance(definition)
//      val inst1 = Instance(definition)
//      inst0.in := in
//      inst1.in := inst0.out
//      out := inst1.out
//    }
//    val top = ChiselGeneratorAnnotation(() => {
//      new Top()
//    }).elaborate
//      .collectFirst { case DesignAnnotation(design: Top) => design }
//      .get
//    intercept[Exception] { Select.collectDeep(top) { case x => x } }
//    intercept[Exception] { Select.getDeep(top)(x => Seq(x)) }
//    intercept[Exception] { Select.instances(top) }
//  }
//
//}
//
//