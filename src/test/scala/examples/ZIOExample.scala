package examples

import chisel3._
import chisel3.internal.Instance
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, DesignAnnotation}
import chiselTests.{ChiselPropSpec, Leaf}
import firrtl.options.Dependency
import firrtl.stage.FirrtlCircuitAnnotation
import zio.{Has, UIO, URIO, ZIO, ZLayer}
import zio.console._

case class Params(a: Int, b: Int, c: Int)

class Simple(params: Params) extends MultiIOModule {
  val in  = IO(Input(UInt(params.a.W)))
  val out = IO(Output(UInt(params.b.W)))
  val leaf = Module(new Leaf)
  out := in + in + params.c.U
}


object Simple extends Utils {
  type SimplePackage = Has[SimplePackage.Service]
  object SimplePackage {
    trait Service {
      def run(): UIO[Simple]
    }
  }

  val default = ZLayer.fromService { params: Params =>
    new SimplePackage.Service {
      override def run() = UIO(build(new Simple(params)))
    }
  }

  def run(): URIO[SimplePackage, Simple] = ZIO.accessM(_.get.run())
}


class Top(simple: Simple) extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))

  val SIMPLE: Simple = Instance(simple)
  val SIMPLE2: Simple = Instance(simple)

  SIMPLE.in := in
  SIMPLE2.in := SIMPLE.out
  out := SIMPLE2.out
}

object Top extends Utils {

  // service binding
  type TopPackage = Has[TopPackage.Service]

  // service declaration
  object TopPackage {
    trait Service {
      def run(): UIO[Top]
    }
  }

  // service implementation
  val default = ZLayer.fromService { simple: Simple.SimplePackage.Service =>
    new TopPackage.Service {
      def run(): UIO[Top] = simple.run().map { s: Simple => build(new Top(s)) }
    }
  }

  // Public accessor
  def run(): URIO[TopPackage, Top] = ZIO.accessM(_.get.run())
}

trait Utils {
  def build[T <: RawModule](gen: => T): T = {
    val stage = new ChiselStage {
      override val targets = Seq( Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.Convert]
      )
    }

    val ret = stage
      .execute(Array("--no-run-firrtl"), Seq(ChiselGeneratorAnnotation(() => gen)))

    println(ret.collectFirst { case FirrtlCircuitAnnotation(cir) => cir.serialize }.get)
    ret.collectFirst { case DesignAnnotation(a) => a } .get.asInstanceOf[T]
  }

}

object BaseSpec extends zio.App {
  def run(args: List[String]) = {
    val paramLayer = ZLayer.succeed(Params(1, 2, 3))
    val liveLayerSimple = paramLayer >>> Simple.default
    val liveLayerTop = liveLayerSimple >>> Top.default
    Top.run().provideCustomLayer(liveLayerTop).exitCode
  }

}

