// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.ChiselAnnotation
import chisel3.internal.InstanceId
import chisel3.internal.firrtl.Circuit
import firrtl.FirrtlExecutionSuccess
import firrtl.annotations.Named
import firrtl.passes.{InlineAnnotation, InlineInstances}
import org.scalatest.{FreeSpec, Matchers}

class AnnotationInline extends FreeSpec with Matchers {

  trait InlineAnnotator {
    self: Module =>

    // This contortion is an attempt to generate a Component annotation from a Module
    case class InlineInstanceId(module: Module) extends InstanceId {
      def instanceName: String = module.instanceName
      def pathName: String = module.pathName
      def parentPathName: String = module.parentPathName
      def parentModName: String = module.parentModName
    }
    def doInlineInstance(module: Module): Unit = {
      annotate(ChiselAnnotation(new InlineInstanceId(module), classOf[InlineInstances], ""))
    }

    // This exercises the other flavor of the FIRRTL API - expressly inline all instances of a Module.
    def doInlineModule(): Unit = {
      annotate(ChiselAnnotation(self, classOf[InlineInstances], ""))
    }
  }

  class AlwaysUsedModule(addAnnos: Boolean) extends Module with InlineAnnotator {
    val io = IO(new Bundle {
      val in = Input(UInt(16.W))
      val out = Output(UInt(16.W))
    })
    io.out := io.in
    if (addAnnos) {
      doInlineModule()
    }
  }

  class MuchUsedModule(addAnnos: Boolean) extends Module {
    val io = IO(new Bundle {
      val in = Input(UInt(16.W))
      val out = Output(UInt(16.W))
    })
    val inlined = Module(new AlwaysUsedModule(addAnnos))
    inlined.io.in := io.in
    io.out := inlined.io.out +% 1.U
  }

  class UsesMuchUsedModule(addAnnos: Boolean) extends Module with InlineAnnotator{
    val io = IO(new Bundle {
      val in = Input(UInt(16.W))
      val out = Output(UInt(16.W))
    })

    val mod0 = Module(new MuchUsedModule(addAnnos))
    val mod1 = Module(new MuchUsedModule(addAnnos))
    val mod2 = Module(new MuchUsedModule(addAnnos))
    val mod3 = Module(new MuchUsedModule(addAnnos))

    mod0.io.in := io.in
    mod1.io.in := mod0.io.out
    mod2.io.in := mod1.io.out
    mod3.io.in := mod2.io.out
    io.out := mod3.io.out

    if(addAnnos) {
      doInlineInstance(mod1)
      doInlineInstance(mod3)
    }
  }

  "Firrtl provides a pass that inlines Module instances" - {
    "Annotations can be added which will invoke this inlining for specific modules instances" in {
      Driver.execute(Array("-X", "low", "--target-dir", "test_run_dir"), () => new UsesMuchUsedModule(addAnnos = true)) match {
        case ChiselExecutionSuccess(_, _, Some(firrtlResult: FirrtlExecutionSuccess)) =>
          val lowFirrtl = firrtlResult.emitted

          lowFirrtl should not include ("inst mod1 ")
          lowFirrtl should not include ("inst mod3 ")
          lowFirrtl should include ("inst mod0 ")
          lowFirrtl should include ("inst mod2 ")
        case _ =>
      }
    }
    "Turning off these annotations does not inline any occurrences" in {
      Driver.execute(Array("-X", "low", "--target-dir", "test_run_dir"), () => new UsesMuchUsedModule(addAnnos = false)) match {
        case ChiselExecutionSuccess(_, _, Some(firrtlResult: FirrtlExecutionSuccess)) =>
          val lowFirrtl = firrtlResult.emitted

          lowFirrtl should include ("inst mod0 ")
          lowFirrtl should include ("inst mod1 ")
          lowFirrtl should include ("inst mod2 ")
          lowFirrtl should include ("inst mod3 ")
        case _ =>
      }
    }
  }
}
