// See LICENSE for license details.

package chiselTests.incremental

import chisel3.testers.BasicTester
import chiselTests.ChiselFlatSpec
import chisel3._
import chisel3.aop.Aspect
import chisel3.internal.firrtl.Circuit
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, ChiselStage, DesignAnnotation}
import firrtl.{EmitCircuitAnnotation, EmittedVerilogCircuit}
import firrtl.stage.FirrtlOption

class IncrementalTester(results: Seq[Int]) extends BasicTester {
  val values = VecInit(results.map(_.U))
  val counter = RegInit(0.U(results.length.W))
  counter := counter + 1.U
  when(counter >= values.length.U) {
    stop()
  }.otherwise {
    when(reset.asBool() === false.B) {
      printf("values(%d) = %d\n", counter, values(counter))
      assert(counter === values(counter))
    }
  }
}

/*
class Foo(fooOpt: Option[Foo]) extends MultiIOModule {
  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))

  val c1Handle = InstanceHandle("c1", fooOpt.get)
  c1Handle { c1 => c1.in }

  val handles = if(fooOpt.nonEmpty) {
    val c1 = InstanceHandle("c1", fooOpt.get)
    c1 { c1 => c1.in }
    val c2 = InstanceHandle("c2", fooOpt.get)
    Seq(c1, c2)
  } else Nil
  if(fooOpt.nonEmpty) {
    handles(0) { _.in := in }
    handles(0).ref{ _.in } := in
    handles(1) { _.in } := handles(0) { _.out }
    handles(0).zip(handles(1)) { (c1, c2) => c1.in := c2.in }
    out := handles(1) { d => d.out }
    out := handles(1) { wrapper => wrapper.get(handles(1)).out }
    handles(1).mirror { d => d.parameters }
    val x = handles(1).materialize()// ref{ d => d.childinstance.mywire }
    handles(1).toAbsoluteTarget(i => i.in.toTarget)
    val stuff = handles(1) { wrapper => wrapper.out }
  } else {
    out := in
  }
}
 */


class IncrementalSpec extends ChiselFlatSpec {
  def elaborate[T <: RawModule](gen: () => T): (T, Circuit) = {
    val annos = ChiselGeneratorAnnotation(gen).elaborate
    val foo: T = annos.collectFirst{case DesignAnnotation(design) => design }.get.asInstanceOf[T]
    val circuit = annos.collectFirst{case ChiselCircuitAnnotation(c) => c }.get
    (foo, circuit)
  }

  def getFirrtl(cir: Circuit): firrtl.ir.Circuit = {
    Aspect.getFirrtl(cir)
  }

  /*
  "Bar" should "elaborate" in {
    val (bar0: Bar, _) = elaborate(() => new Bar(None))
    val (bar1: Bar, _) = elaborate(() => new Bar(Some(bar0)))
    val x = bar1.handles(0) { bar0 => bar0.toAbsoluteTarget }
    println(x.serialize)
  }

   */
  /*
  "Foo should elaborate" should "in" in {
    class Foo(fooOpt: Option[Foo]) extends MultiIOModule {
      val in = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      val handles = if(fooOpt.nonEmpty) {
        val c1 = InstanceHandle("c1", fooOpt.get)
        val c2 = InstanceHandle("c2", fooOpt.get)
        Seq(c1, c2)
      } else Nil
    }
    val (foo0: Foo, cir0) = elaborate(() => new Foo(None))
    val (foo1: Foo, cir1) = elaborate(() => new Foo(Some(foo0)))
    //val x = foo1.handles(0).toAbsoluteTarget {
    //  foo0 => foo0.in.toAbsoluteTarget
    //}
    //println(x.serialize)
    println(getFirrtl(cir1).serialize)
  }

   */

  "Parent references to instances" should "elaborate without errors" in { }
  "Parent connections to/from instances" should "elaborate without errors" in { }
  "Instance references" should "serialize to specified name" in {}
  "Absolute targets post elaboration" should "build correct target" in {}
  "Relative targets post elaboration" should "build correct target" in {}
  "Original modules" should "be possible to garbage collect" in {}
}


/** Plan:
  *
  * 1) Remove all child -> parent direct references
  * 2) Implement Stashes as input+output of Builder
  * 3) Deprecate or support old serialization path
  * 4)
  *
  */
