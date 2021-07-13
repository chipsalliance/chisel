// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.experimental.annotate
import chisel3.internal.{interface, public}
import _root_.firrtl.annotations._

object InstanceSpec {
  object Examples {
    @interface
    class AddOne(hasInner: Boolean) extends MultiIOModule {
      @public val in  = IO(Input(UInt(32.W)))
      @public val out = IO(Output(UInt(32.W)))
      @public val innerWire = if(hasInner) Some(Wire(UInt(32.W))) else None
      innerWire match {
        case Some(w) =>
          w := in + 1.U
          out := w
        case None =>
          out := in + 1.U
      }
    }

    //implicit class AddOneHandle(i: Instance[AddOne]) {
    //  val in = i.d(_.in)
    //  val out = i.d(_.out)
    //  val innerWire = i.d(_.innerWire)
    //}
  
    @interface
    class AddTwo extends MultiIOModule {
      @public val in  = IO(Input(UInt(32.W)))
      @public val out = IO(Output(UInt(32.W)))
      val template = Template(new AddOne(true))
      @public val i0 = Instance(template)
      @public val i1 = Module(new AddOne(true))
      //@public val x = 10 //ERRORS!!!
      i0.in := in
      i1.in := i0.out
      out := i1.out
    }
    //implicit class AddTwoHandle(i: Instance[AddTwo]) {
    //  def in = i.d(_.in)
    //  def out = i.d(_.out)
    //  def i0 = i.i(_.i0)
    //  def i1 = i.m(_.i1)
    //}

    //class AddNoneOrTwo(addTwo: Boolean) extends MultiIOModule {
    //  val in  = IO(Input(UInt(32.W)))
    //  val out = IO(Output(UInt(32.W)))
    //  val template = Template(new AddOne)
    //  val i0Opt = if(addTwo) Some(Instance(template)) else None
    //  val i1Opt = if(addTwo) Some(Module(new AddOne)) else None
    //  i1Opt.zip(i0Opt) match {
    //    case (i0, i1) =>
    //      i0.in := in
    //      i1.in := i0.out
    //      out := i1.out
    //    case _ =>
    //      out := in
    //  }
    //}
    //implicit class AddNoneOrTwoHandle(i: Instance[AddNoneOrTwo]) {
    //  def in = i(_.in)
    //  def out = i(_.out)
    //  def i0Opt = i(_.i0Opt)
    //  def i1Opt = i(_.i1Opt)
    //}


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
}


class InstanceSpec extends ChiselFlatSpec with Utils {
  import InstanceSpec.Examples._
  "Template/Instance" should "enable instantiating the same instance multiple times" in {
    import InstanceSpec.Annotations._
    class AddOneTester extends BasicTester {
      val template: Template[AddOne] = Template(new AddOne(true))
      val i0: Instance[AddOne] = Instance(template)
      val i1: Instance[AddOne] = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      i0.innerWire.map(x => mark(x, "Jack Was Here"))
      chisel3.assert(i1.out === 44.U)
      stop()
    }
    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddOne>innerWire").asInstanceOf[ReferenceTarget], "Jack Was Here"))
    assertTesterPasses(new AddOneTester)
  }
  "Template/Instance" should "enable instantiating nestingly, with modules" in {
    import InstanceSpec.Annotations._
    class AddOneTester extends BasicTester {
      val template = Template(new AddTwo)
      val i0 = Instance(template)
      val i1 = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      i0.i0.innerWire.map(x => mark(x, "Adam Was Here"))
      i0.i1.innerWire.map(x => mark(x, "Megan Was Here"))
      chisel3.assert(i1.out === 46.U)
      stop()
    }

    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddTwo/i0:AddOne>innerWire").asInstanceOf[ReferenceTarget], "Adam Was Here"))
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddTwo/i1:AddOne_2>innerWire").asInstanceOf[ReferenceTarget], "Megan Was Here"))
    assertTesterPasses(new AddOneTester)
  }
  "Template/Instance" should "implicitly convert modules to instances" in {
    import InstanceSpec.Annotations._
    def wireUp(i0: Instance[AddOne], i1: Instance[AddOne]): Unit = {
      i1.in := i0.out
    }
    class AddOneTester extends BasicTester {
      val template = Template(new AddOne(false))
      val i0 = Instance(template)
      val i1 = Module(new AddOne(true))
      i0.in := 42.U
      wireUp(i0, i1)
      i1.in := i0.out
      chisel3.assert(i1.out === 44.U)
      stop()
    }

    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
    assertTesterPasses(new AddOneTester)
  }
}
