// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util.experimental.{InlineInstance, FlattenInstance}
import chisel3.internal.firrtl.Circuit
import firrtl.FirrtlExecutionSuccess
import firrtl.passes.InlineAnnotation
import firrtl.annotations.Annotation
import firrtl.transforms.FlattenAnnotation
import firrtl.analyses.InstanceGraph
import firrtl.{ir => fir}
import firrtl.WDefInstance
import firrtl.Mappers._
import org.scalatest.{FreeSpec, Matchers}

class InlineSpec extends FreeSpec with ChiselRunners with Matchers {

  trait Internals { this: Module =>
    val io = IO(new Bundle{ val a = Input(Bool()) })
  }
  class Sub extends Module with Internals
  trait HasSub { this: Module with Internals =>
    val sub = Module(new Sub)
    sub.io.a := io.a
  }

  class Foo extends Module with Internals with InlineInstance with HasSub
  class Bar extends Module with Internals with HasSub
  class Baz extends Module with Internals with HasSub
  class Qux extends Module with Internals with HasSub

  def collectInstances(c: fir.Circuit, top: Option[String] = None): Seq[String] = new InstanceGraph(c)
    .fullHierarchy.values.flatten.toSeq
    .map( v => (top.getOrElse(v.head.name) +: v.tail.map(_.name)).mkString(".") )

  "Module Inlining" - {
    class Top extends Module with Internals {
      val x = Module(new Foo)
      val y = Module(new Bar with InlineInstance)
      val z = Module(new Bar)
      Seq(x, y, z).map(_.io.a := io.a)
    }
    "should compile to low FIRRTL" - {
      Driver.execute(Array("-X", "low", "--target-dir", "test_run_dir"), () => new Top) match {
        case ChiselExecutionSuccess(Some(chiselCircuit), _, Some(firrtlResult: FirrtlExecutionSuccess)) =>
          "emitting TWO InlineAnnotation at the CHIRRTL level" in {
            chiselCircuit.annotations.map(_.toFirrtl).collect{ case a: InlineAnnotation => a }.size should be (2)
          }
          "low FIRRTL should contain only instance z" in {
            val instances = collectInstances(firrtlResult.circuitState.circuit, Some("Top")).toSet
            Set("Top", "Top.x_sub", "Top.y_sub", "Top.z", "Top.z.sub") should be (instances)
          }
      }
    }
  }

  "Module Flattening" - {
    class Top extends Module with Internals {
      val x = Module(new Qux with FlattenInstance)
      x.io.a := io.a
    }
    "should compile to low FIRRTL" - {
      Driver.execute(Array("-X", "low", "--target-dir", "test_run_dir"), () => new Top) match {
        case ChiselExecutionSuccess(Some(chiselCircuit), chirrtl, Some(firrtlResult: FirrtlExecutionSuccess)) =>
          "emitting ONE FlattenAnnotation at the CHIRRTL level" in {
            chiselCircuit.annotations.map(_.toFirrtl).collect{ case a: FlattenAnnotation => a }.size should be (1)
          }
          "low FIRRTL should contain instance x only" in {
            val instances = collectInstances(firrtlResult.circuitState.circuit, Some("Top")).toSet
            Set("Top", "Top.x") should be (instances)
          }
      }
    }
  }
}
