// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.experimental.dataview._
import chisel3.aop.Select
import chisel3.experimental.annotate


class AddOne extends Module with ViewableAs[AddOne, PipeInterface] {
  implicit val moduleView: DataView[AddOne, PipeInterface] = DataView(_.in -> _.in, _.out -> _.out)
  val in  = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  val innerWire = Wire(UInt(32.W))
  innerWire := in + 1.U
  out := innerWire
}
class AddTwo extends Module with ViewableAs[AddTwo, PipeInterface] {
  implicit val moduleView: DataView[AddTwo, PipeInterface] = DataView(_.in -> _.in, _.out -> _.out)
  val in  = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  val template = Template(new AddOne)
  val inst0 = Instance(template)
  inst0(_.in) := in
  val inst1 = Instance(template)
  inst1(_.in) := inst0(_.out)
  out := inst1(_.out)
}
class Piper[T <: Module with ViewableAs[T, PipeInterface]](template: T) extends Module with ViewableAs[Piper[T], PipeInterface] {
  implicit val moduleView: DataView[Piper[T], PipeInterface] = DataView(_.in -> _.in, _.out -> _.out)

  val in  = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))
  val interface = new PipeInterface
  val inst0 = Instance(template).viewAs(interface)(template.instanceView)
  inst0.in := in
  val inst1 = Instance(template).viewAs(interface)(template.instanceView)
  inst1.in := inst0.out
  out := inst1.out
}
class PipeInterface extends Bundle {
  val in = UInt(32.W)
  val out = UInt(32.W)
}

object MyAnnotation {
  import _root_.firrtl.annotations._
  case class MyFirrtlAnnotation(target: ReferenceTarget, tag: String) extends SingleTargetAnnotation[ReferenceTarget] {
    def duplicate(n: ReferenceTarget): Annotation = this.copy(target = n)
  }
  case class MyAnnotation(d: Data, tag: String) extends chisel3.experimental.ChiselAnnotation {
    def toFirrtl = MyFirrtlAnnotation(d.toTarget, tag)
  }
  implicit class Marker(d: Data) {
    def mark(tag: String): Unit = annotate(MyAnnotation(d, tag))
  }
}

class TemplateSpec extends ChiselFlatSpec with Utils {
  import Instance._
  "Template/Instance" should "enable instantiating the same instance multiple times" in {
    class AddOneTester extends BasicTester {
      val template = Template(new AddOne)
      val i0 = Instance(template).viewAs(new PipeInterface)(template.instanceView)
      val i1 = Instance(template).viewAs(new PipeInterface)(template.instanceView)
      i0.in := 42.U
      i1.in := i0.out
      chisel3.assert(i1.out === 44.U)
      stop()
    }
    println((new ChiselStage).emitChirrtl(gen = new AddOneTester, args = Array("--full-stacktrace")))
    assertTesterPasses(new AddOneTester)
  }
  "Template/Instance" should "enable connecting module and instance ports with the same code (using views)" in {
    class AddOneTester extends BasicTester {
      val template = Template(new AddOne)
      val i0 = Module(new AddOne).viewAs(new PipeInterface)(template.moduleView)
      val i1 = Instance(template).viewAs(new PipeInterface)(template.instanceView)
      i0.in := 42.U
      i1.in := i0.out
      chisel3.assert(i1.out === 44.U)
      stop()
    }
    println((new ChiselStage).emitChirrtl(gen = new AddOneTester))
    assertTesterPasses(new AddOneTester)
  }
  "Template/Instance" should "work recursively" in {
    class RecursiveTester extends BasicTester {
      val addOne = Template(new AddOne)
      val addTwo = Template(new Piper(addOne))
      val addFour = Template(new Piper(addTwo))
      val addEight = Template(new Piper(addFour))
      val inst = Instance(addEight).viewAs(new PipeInterface)(addEight.instanceView)
      inst.in := 42.U
      chisel3.assert(inst.out === 50.U)
      stop()
    }
    println((new ChiselStage).emitChirrtl(gen = new RecursiveTester))
    assertTesterPasses(new RecursiveTester)
  }
  "Template/Instance" should "let annotating an instance" in {
    import MyAnnotation._
    class RecursiveTester extends BasicTester {
      val addTwo = Template(new AddTwo)
      val handle = Instance(addTwo)
      handle(_.in) := 42.U

      // Recursive instance accessing
      handle(_.inst0(_.innerWire.mark("Instance0")))
      
      // Iterative instance accessing
      handle(_.inst1)(_.innerWire).mark("Instance1")

      chisel3.assert(handle(_.out) === 44.U)
      stop()
    }
    println((new ChiselStage).emitChirrtlWithAnnotations(gen = new RecursiveTester))
    println((new ChiselStage).emitVerilog(gen = new RecursiveTester))
    assertTesterPasses(new RecursiveTester)
  }
  /*
  Thoughts:
   - We have different semantics calling toTarget through an Instance, than through a Module.
   - Consider different bindings, e.g. CMR binding?

   - summon instance, use macros to define extension methods on instance for each module member. call clonetype on any returned data, generating an XMR for non-ports
   - 
  */
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
