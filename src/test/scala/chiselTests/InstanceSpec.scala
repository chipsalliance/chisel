// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.experimental.annotate
import _root_.firrtl.annotations._
import chisel3.experimental.dataview._

object InstanceSpec{
  object Examples {
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

    implicit class AddOneHandle(i: Instance[AddOne]) {
      val in = i(_.in)
      val out = i(_.out)
      val innerWire = i(_.innerWire)
    }

    implicit class AddTwoHandle(i: Instance[AddTwo]) {
      def in = i(_.in)
      def out = i(_.out)
      def i0 = i(_.i0)
      def i1 = i(_.i1)
    }
  }
  object Annotations {
    case class MarkAnnotation(target: ReferenceTarget, tag: String) extends SingleTargetAnnotation[ReferenceTarget] {
      def duplicate(n: ReferenceTarget): Annotation = this.copy(target = n)
    }
    case class MarkChiselAnnotation(d: Data, tag: String) extends chisel3.experimental.ChiselAnnotation {
      def toFirrtl = MarkAnnotation(d.toTarget, tag)
    }
    def mark(d: Data, tag: String): Unit = annotate(MarkChiselAnnotation(d, tag))
  }

  object Views {
    import Examples._
    object Pipe {
      class PipeInterface extends Bundle {
        val in = UInt(32.W)
        val out = UInt(32.W)
      }
      import chiselTests.experimental.HWTuple._
      implicit val view = DataView[HWTuple2[UInt, UInt], PipeInterface](_._1 -> _.in, _._2 -> _.out)

      def pipeEm(pipes: Seq[PipeInterface]): PipeInterface = {
        require(pipes.nonEmpty)
        val out = pipes.tail.foldLeft(pipes.head.out) { case (source, pipe) =>
          pipe.in := source
          pipe.out
        }
        new HWTuple2(pipes.head.in, out).viewAs(new PipeInterface)
      }
      implicit val addonemoduleview = DataView[AddOne, PipeInterface](_.in -> _.in, _.out -> _.out)
      implicit val addoneinstanceview = DataView[Instance[AddOne], PipeInterface](_.in -> _.in, _.out -> _.out)
    }
    object ClearPipe {
      class ClearPipeInterface extends Bundle {
        val in = UInt(32.W)
        val innerWire = UInt(32.W)
        val out = UInt(32.W)
      }
      import chiselTests.experimental.HWTuple._
      implicit val view = DataView[HWTuple3[UInt, UInt, UInt], ClearPipeInterface](_._1 -> _.in, _._2 -> _.innerWire, _._3 -> _.out)

      implicit val addonemoduleview = DataView[AddOne, ClearPipeInterface](_.in -> _.in, _.innerWire -> _.innerWire, _.out -> _.out)
      implicit val addoneinstanceview = DataView[Instance[AddOne], ClearPipeInterface](_.in -> _.in, _.innerWire -> _.innerWire, _.out -> _.out)
      
    }
  }
}


class InstanceSpec extends ChiselFreeSpec with Utils {
  import InstanceSpec.Examples._
  "Template/Instance should enable instantiating the same instance multiple times" - {
    import InstanceSpec.Annotations._
    class AddOneTester extends BasicTester {
      val template = Template(new AddOne)
      val i0 = Instance(template)
      val i1 = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      mark(i0.innerWire, "Jack Was Here")
      chisel3.assert(i1.out === 44.U)
      stop()
    }
    "Annotating only one signal in one instance" in {
      val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
      annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddOne>innerWire").asInstanceOf[ReferenceTarget], "Jack Was Here"))
    }
    "Still pass the simulation" in {
      assertTesterPasses(new AddOneTester)
    }
  }
  "Template/Instance should enable instantiating nestingly" - {
    import InstanceSpec.Annotations._
    class AddOneTester extends BasicTester {
      val template = Template(new AddTwo)
      val i0 = Instance(template)
      val i1 = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      mark(i0.i1.innerWire, "Megan Was Here")
      chisel3.assert(i1.out === 46.U)
      stop()
    }
    "Annotating only one signal in a doubly nested instance" in {
      val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
      annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddTwo/i1:AddOne>innerWire").asInstanceOf[ReferenceTarget], "Megan Was Here"))
    }
    "Still pass the simulation" in {
      assertTesterPasses(new AddOneTester)
    }
  }
  "Template/Instance, as well as modules, should be viewable as a bundle" - {
    import InstanceSpec.Views.Pipe._
    class AddOneTester extends BasicTester {
      val template = Template(new AddOne)
      val instances = Seq.fill(2)(Instance(template).viewAs(new PipeInterface))
      val modules = Seq.fill(2)(Module(new AddOne).viewAs(new PipeInterface))
      val pipe = pipeEm(instances ++ modules)
      pipe.in := 42.U
      chisel3.assert(pipe.out === 46.U)
      stop()
    }
    "Still pass the simulation" in {
      assertTesterPasses(new AddOneTester)
    }
  }
  "Template/Instance should be able to annotate an internal field of a viewed instance and module" - {
    import InstanceSpec.Annotations._
    import InstanceSpec.Views.ClearPipe._
    class AddOneTester extends BasicTester {
      val template = Template(new AddOne)
      val i0 = Module(new AddOne).viewAs(new ClearPipeInterface)
      val i1 = Instance(template).viewAs(new ClearPipeInterface)
      i0.in := 42.U
      i1.in := i0.out
      mark(i0.innerWire, "This is a module")
      mark(i1.innerWire, "This is an instance")
      chisel3.assert(i1.out === 44.U)
      stop()
    }
    "Annotating only one signal in a doubly nested instance" in {
      val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
      println(output)
      annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddOne>innerWire").asInstanceOf[ReferenceTarget], "This is a module"))
      annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i1:AddOne>innerWire").asInstanceOf[ReferenceTarget], "This is an instance"))
    }
    "Still pass the simulation" in {
      assertTesterPasses(new AddOneTester)
    }

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
