// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform, annotate}
import chisel3.internal.InstanceId
import chisel3.testers.BasicTester
import firrtl.{CircuitForm, CircuitState, LowForm, Transform}
import firrtl.annotations._
import org.scalatest._

class ToTargetSpec extends FlatSpec with Matchers {

  case class DummyAnnotation(target: Target, value: String) extends SingleTargetAnnotation[Target] {
    def duplicate(n: Target): DummyAnnotation = this.copy(target = n)
  }

  def dummy(component: InstanceId, value: String): Unit = annotate(new ChiselAnnotation {
    def toFirrtl: DummyAnnotation = DummyAnnotation(component.toTarget, value)
  })

  def dummyAbsolute(component: InstanceId, value: String): Unit = annotate(new ChiselAnnotation {
    def toFirrtl: DummyAnnotation = DummyAnnotation(component.toAbsoluteTarget, value)
  })

  class ModA extends Module {

    override def desiredName = "ModA"

    val io = IO(new Bundle {
      val in = Input(UInt(32.W))
      val out = Output(UInt(32.W))
    })

    val modB = Module(new ModB)
    modB.io.in := io.in
    io.out := modB.io.out
  }

  class ModB extends Module {

    override def desiredName = "ModB"

    val io = IO(new Bundle {
      val in = Input(UInt(32.W))
      val out = Output(UInt(32.W))
    })

    val x = Reg(new Bundle {
      val y = UInt(32.W)
    })

    x.y := io.in
    io.out := x.y

    dummyAbsolute(this, s"absolute ModB")
    dummy(this, s"ModB")

    dummyAbsolute(x, s"absolute ModB.x")
    dummy(x, s"ModB.x")

    dummyAbsolute(x.y, s"absolute ModB.x.y")
    dummy(x.y, s"ModB.x.y")
  }

  class Top extends Module {

    override def desiredName = "Top"

    val io = IO(new Bundle {
      val in   = Input(UInt(32.W))
      val out  = Output(UInt(32.W))
    })

    val modA = Module(new ModA)

    modA.io.in := io.in
    io.out := modA.io.out
  }

  "toTarget and toAbsoluteTarget" should "create correct targets" in {
    Driver.execute(Array("--target-dir", "test_run_dir"), () => new Top) match {
      case ChiselExecutionSuccess(Some(circuit), emitted, _) =>
        val annos = circuit.annotations.map(_.toFirrtl)
        annos.count(_.isInstanceOf[DummyAnnotation]) should be (6)

        val modB = ModuleTarget("Top", "ModB")
        val modB_x = modB.ref("x")
        val modB_x_y = modB_x.field("y")

        val top = ModuleTarget("Top", "Top")
        val instB = top.instOf("modA", "ModA").instOf("modB", "ModB")
        val instB_x = instB.ref("x")
        val instB_x_y = instB_x.field("y")

        annos.count {
          case DummyAnnotation(target, "ModB") if target == modB => true
          case _ => false
        } should be (1)

        annos.count {
          case DummyAnnotation(target, "ModB.x") if target == modB_x => true
          case _ => false
        } should be (1)

        annos.count {
          case DummyAnnotation(target, "ModB.x.y") if target == modB_x_y => true
          case _ => false
        } should be (1)

        annos.count {
          case DummyAnnotation(target, "absolute ModB") if target == instB => true
          case _ => false
        } should be (1)

        annos.count {
          case DummyAnnotation(target, "absolute ModB.x") if target == instB_x => true
          case _ => false
        } should be (1)

        annos.count {
          case DummyAnnotation(target, "absolute ModB.x.y") if target == instB_x_y => true
          case _ => false
        } should be (1)

      case _ =>
        assert(false)
    }
  }
}
