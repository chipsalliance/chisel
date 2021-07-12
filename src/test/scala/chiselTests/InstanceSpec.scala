// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.experimental.annotate
import _root_.firrtl.annotations._

object InstanceSpec {
  object Examples {
    class AddOne extends MultiIOModule {
      val in  = IO(Input(UInt(32.W)))
      val out = IO(Output(UInt(32.W)))
      val innerWire = Wire(UInt(32.W))
      innerWire := in + 1.U
      out := innerWire
    }
  
    class AddTwo extends MultiIOModule {
      val in  = IO(Input(UInt(32.W)))
      val out = IO(Output(UInt(32.W)))
      val template = Template(new AddOne)
      val i0 = Instance(template)
      val i1 = Module(new AddOne)
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
}


class InstanceSpec extends ChiselFlatSpec with Utils {
  import InstanceSpec.Examples._
  "Template/Instance" should "enable instantiating the same instance multiple times" in {
    import InstanceSpec.Annotations._
    class AddOneTester extends BasicTester {
      val template: AddOne = Template(new AddOne)
      val i0: Instance[AddOne] = Instance(template)
      val i1: Instance[AddOne] = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      mark(i0.innerWire, "Jack Was Here")
      chisel3.assert(i1.out === 44.U)
      stop()
    }
    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddOne>innerWire").asInstanceOf[ReferenceTarget], "Jack Was Here"))
    assertTesterPasses(new AddOneTester)
  }
  "Template/Instance" should "enable instantiating nestingly" in {
    import InstanceSpec.Annotations._
    class AddOneTester extends BasicTester {
      val template = Template(new AddTwo)
      val i0 = Instance(template)
      val i1 = Instance(template)
      i0.in := 42.U
      i1.in := i0.out
      mark(i0.i0.innerWire, "Adam Was Here")
      mark(i0.i1.innerWire, "Megan Was Here")
      chisel3.assert(i1.out === 46.U)
      stop()
    }

    val (output, annotations) = (new ChiselStage).emitChirrtlWithAnnotations(gen = new AddOneTester, args = Array("--full-stacktrace"))
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddTwo/i0:AddOne>innerWire").asInstanceOf[ReferenceTarget], "Adam Was Here"))
    annotations.toSeq should contain (MarkAnnotation(Target.deserialize("~AddOneTester|AddOneTester/i0:AddTwo/i1:AddOne_2>innerWire").asInstanceOf[ReferenceTarget], "Megan Was Here"))
    assertTesterPasses(new AddOneTester)
  }
}
