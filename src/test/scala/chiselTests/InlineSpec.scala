// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util.experimental.{InlineInstance, FlattenInstance}
import firrtl.FirrtlExecutionSuccess
import firrtl.passes.InlineAnnotation
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlStage}
import firrtl.transforms.FlattenAnnotation
import firrtl.analyses.InstanceGraph
import firrtl.{ir => fir}
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

class InlineSpec extends AnyFreeSpec with ChiselRunners with Matchers {

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

  val chiselStage = new ChiselStage
  val firrtlStage = new FirrtlStage

  "Module Inlining" - {
    class Top extends Module with Internals {
      val x = Module(new Foo)
      val y = Module(new Bar with InlineInstance)
      val z = Module(new Bar)
      Seq(x, y, z).map(_.io.a := io.a)
    }
    "should compile to low FIRRTL" - {
      val chiselAnnotations =
        chiselStage
          .execute(Array("--no-run-firrtl", "--target-dir", "test_run_dir"),
                   Seq(ChiselGeneratorAnnotation(() => new Top)))

      chiselAnnotations.collect{ case a: InlineAnnotation => a } should have length (2)

      val instanceNames =
        firrtlStage
          .execute(Array("-X", "low"), chiselAnnotations)
          .collectFirst {
            case FirrtlCircuitAnnotation(circuit) => circuit
          }.map(collectInstances(_, Some("Top")))
          .getOrElse(fail)

      instanceNames should contain theSameElementsAs Set("Top", "Top.x_sub", "Top.y_sub", "Top.z", "Top.z.sub")
    }
  }

  "Module Flattening" - {
    class Top extends Module with Internals {
      val x = Module(new Qux with FlattenInstance)
      x.io.a := io.a
    }
    "should compile to low FIRRTL" - {
      val chiselAnnotations =
        chiselStage
          .execute(Array("-X", "low", "--target-dir", "test_run_dir"),
                   Seq(ChiselGeneratorAnnotation(() => new Top)))

      chiselAnnotations.collect{ case a: FlattenAnnotation => a} should have length(1)

      val instanceNames =
        firrtlStage
          .execute(Array("-X", "low"), chiselAnnotations)
          .collectFirst {
            case FirrtlCircuitAnnotation(circuit) => circuit
          }.map(collectInstances(_, Some("Top")))
          .getOrElse(fail)

      instanceNames should contain theSameElementsAs Set("Top", "Top.x")
    }
  }
}
