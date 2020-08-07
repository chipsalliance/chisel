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

class SimpleX(int: Int) extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  //val in  = chisel3.experimental.isInstance.check(IO(Input(UInt(3.W))))
  val out = IO(Output(UInt(3.W)))
  //val out = chisel3.experimental.isInstance.check(IO(Output(UInt(3.W))))
  val leaf = Module(new Leaf)
  //val leaf = chisel3.experimental.isInstance.check(Module(new Leaf))
  out := in + in
  //chisel3.experimental.isInstance.check(() => out := in + in)
  val x = int
  //val x = chisel3.experimental.isInstance.check(int)
}


class Top(simple: SimpleX) extends MultiIOModule {
  val in  = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))

  // 1) Original backing module of type Simple
  // 2) New Black Box module
  // 3) New Empty Module of type Simple

  val SIMPLE: SimpleX = Instance(simple)
  val SIMPLE2: SimpleX = Instance(simple)

  SIMPLE.in := in
  //SIMPLE.useInstance(SIMPLE.getBackingModule[SimpleX].in) := in

  SIMPLE2.in := SIMPLE.out

  out:= SIMPLE2.out
  //out := SIMPLE2.useInstance(SIMPLE.getBackingModule[SimpleX].out)
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

  property("Explicit example test case") {
    //Diplomacy occurs

    //Chisel Construction
    val simple: SimpleX = build { new SimpleX(10) }
    val top: Top = build { new Top(simple) }
  }

}
//Prints out:
/*
[info] [0.002] Elaborating design...
[info] [0.063] Done elaborating.
circuit SimpleX :
  module Leaf :
    input clock : Clock
    input reset : Reset
    input in : UInt<3>
    output out : UInt<3>

    node _out_T = add(in, in) @[InstanceSpec.scala 24:13]
    node _out_T_1 = tail(_out_T, 1) @[InstanceSpec.scala 24:13]
    out <= _out_T_1 @[InstanceSpec.scala 24:7]

  module SimpleX :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<3>
    output out : UInt<3>

    inst leaf of Leaf @[InstanceSpec.scala 33:20]
    leaf.clock <= clock
    leaf.reset <= reset
    node _out_T = add(in, in) @[InstanceSpec.scala 35:13]
    node _out_T_1 = tail(_out_T, 1) @[InstanceSpec.scala 35:13]
    out <= _out_T_1 @[InstanceSpec.scala 35:7]

[info] [0.000] Elaborating design...
[info] [0.024] Done elaborating.
circuit Top :
  extmodule SimpleX :
    output out : UInt<3>
    input in : UInt<3>
    input reset : UInt<1>
    input clock : Clock

    defname = SimpleX


  extmodule SimpleX_2 :
    output out : UInt<3>
    input in : UInt<3>
    input reset : UInt<1>
    input clock : Clock

    defname = SimpleX


  module Top :
    input clock : Clock
    input reset : UInt<1>
    input in : UInt<3>
    output out : UInt<3>

    inst SIMPLE of SimpleX @[InstanceSpec.scala 50:33]
    SIMPLE.clock is invalid
    SIMPLE.reset is invalid
    SIMPLE.in is invalid
    SIMPLE.out is invalid
    inst SIMPLE2 of SimpleX_2 @[InstanceSpec.scala 51:34]
    SIMPLE2.clock is invalid
    SIMPLE2.reset is invalid
    SIMPLE2.in is invalid
    SIMPLE2.out is invalid
    SIMPLE.in <= in @[InstanceSpec.scala 53:13]
    SIMPLE2.in <= SIMPLE.out @[InstanceSpec.scala 56:14]
    out <= SIMPLE2.out @[InstanceSpec.scala 58:6]
 */
