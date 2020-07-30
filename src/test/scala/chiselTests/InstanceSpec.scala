package chiselTests

import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage, DesignAnnotation}
import chisel3._
import chisel3.internal.Instance
import firrtl.options.Dependency
import firrtl.stage.FirrtlCircuitAnnotation

class Leaf extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  out := in + in
  val x = 10
}

class Simple extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  val leaf = Module(new Leaf)
  out := in + in
  val x = 10
}

class TopExplicit(simple: Simple) extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))

  //val SIMPLE: Simple = Instance(simple)
  val SIMPLE: Simple = Instance.createInstance(simple, Some("SIMPLE"))

  //SIMPLE.in := in
  SIMPLE.useInstance("SIMPLE")(SIMPLE.in) := in

  //out := SIMPLE.out
  out := SIMPLE.useInstance("SIMPLE")(SIMPLE.out)
}

class TopImplicit(simple: Simple) extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))

  val SIMPLE: Simple = Instance(simple)
  val SIMPLE2: Simple = Instance(simple)

  SIMPLE.in := in
  SIMPLE2.in := SIMPLE.out
  out := SIMPLE2.out

}

class InstanceSpec extends ChiselPropSpec with Utils {
  /** Return a Chisel circuit for a Chisel module
    * @param gen a call-by-name Chisel module
    */
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

  property("First example test case") {
    val simple: Simple = build { new Simple }
    val top: TopImplicit = build { new TopImplicit(simple) }
  }

  property("Explicit example test case") {
    val simple: Simple = build { new Simple }
    val top: TopExplicit = build { new TopExplicit(simple) }
  }

}
