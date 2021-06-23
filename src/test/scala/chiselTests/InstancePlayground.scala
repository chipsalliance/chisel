// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.experimental.dataview._
import chisel3.aop.Select
import chisel3.experimental.annotate

object PlaygroundExamples {
  class AddOne extends Module {
    val in  = IO(Input(UInt(32.W)))
    val out = IO(Output(UInt(32.W)))
    val innerWire = Wire(UInt(32.W))
    innerWire := in + 1.U
    out := innerWire
  }
  
  class AddTwo extends Module {
    val in  = IO(Input(UInt(32.W)))
    val out = IO(Output(UInt(32.W)))
    val template = Template(new AddOne)
    val i0 = Instance(template)
    val i1 = Instance(template)
    i0.in := in
    i1.in := i0.out
    out := i1.out
  }
  implicit class AddJuanHandle(i: Instance[AddOne]) {
    val in = i(_.in)
    val out = i(_.out)
    val innerWire = i(_.innerWire)
  }
  implicit class AddTwwoHandle(i: Instance[AddTwo]) {
    def in = i(_.in)
    def out = i(_.out)
    def i0 = i(_.i0)
    def i1 = i(_.i1)
  }
}

class InstancePlayground extends ChiselFlatSpec with Utils {
  import Instance._
  import PlaygroundExamples._
  "Template/Instance" should "enable instantiating the same instance multiple times" in {
    class AddOneTester extends BasicTester {
      val template = Template(new AddOne)
      val i0 = Instance(template)
      val i1 = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      dontTouch(i0.innerWire)
      chisel3.assert(i1.out === 44.U)
      stop()
    }
    println((new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace")))
    assertTesterPasses(new AddOneTester)
  }
  "Template/Instance" should "enable instantiating nestingly" in {
    class AddOneTester extends BasicTester {
      val template = Template(new AddTwo)
      val i0 = Instance(template)
      val i1 = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      dontTouch(i0.i1.innerWire)
      chisel3.assert(i1.out === 46.U)
      stop()
    }
    println((new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace")))
    assertTesterPasses(new AddOneTester)
  }
  "Template/Instance" should "be viewable as bundle, same as module" in {}
  "Template/Instance" should "be able to refactored to extend same interface as module" in {

  }
  //{
  //  // Option 1
  //  val pc = core(_.cache)(_.pc)
  //  annotate(pc)

  //  // Option 2
  //  // I like option 2 better
  //  val pc = core(_.cache)(x => annotate(x.pc))

  //  
  //}
  //"Template/Instance" should "be able to get instance targets from templates, for use in aspects and chisel annotations" in {
    //import chisel3.aop.injecting._
    //import chisel3.aop._
    //val plus2aspect = InjectingAspect(
    //  {dut: AddOneTester => Select.collectDeep(dut, InstanceContext(Seq(dut))) {
    //    case (t: AddOne, context) =>
    //      println(context.toTarget)
    //      println(s"HERE: ${t.in.absoluteTarget(context)}")
    //      t
    //  }},
    //  {dut: AddOne =>
    //    println("HERE")
    //    dut.out := dut.in + 1.asUInt
    //  }
    //)
    //println((new ChiselStage).emitChirrtl(gen = new AddOneTester))
    //assertTesterPasses( new AddOneTester, Nil, Seq(plus2aspect))
  //}
  //"Template/Instance" should "work with grandcentral" { ??? }
}
