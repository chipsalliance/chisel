// See LICENSE.txt for details


package ciclTests


import chiselTests.ChiselPropSpec
import chisel3._
import chisel3.core.dontTouch
import chisel3.experimental.MultiIOModule
import chisel3.libs.BreakPoint.BreakPointAnnotation
import chisel3.libs.aspect.ModuleAspect
import chisel3.libs.{AssertDelay, BreakPoint, CMR}
import firrtl.annotations._
import firrtl.ir.{Input => _, Module => _, Output => _, _}
import firrtl.passes.ToWorkingIR
import firrtl.passes.wiring.WiringInfo

import scala.collection.mutable

/*
class B extends MultiIOModule {
  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  val a1 = Module(new A)//.suggestName("blah")
  a1.in := in
  val a2 = Module(new A)
  a2.in := a1.out
  out := a2.out

  def addBreakpoints = BreakPoint("bpB", this, (b: B, cmr: CMR) => {
    val breakReg = RegNext(cmr(b.a1.myreg)).suggestName("breakReg")
    breakReg > 10.U && cmr(b.a1.myreg) < 20.U
  })
}
*/
class Buffer(delay: Int) extends MultiIOModule {
  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  val regs = Reg(t=Vec(delay, UInt(3.W)))
  out := regs.foldLeft(in){ (source, r) =>
    r := source
    r
  }
}

class CICLSpec extends ChiselPropSpec {
  private val ModuleRegex = """\s*module\s+(\w+)\b.*""".r
  def countModules(verilog: String): Int =
    (verilog split "\n"  collect { case ModuleRegex(name) => name }).size

  /**
    * Limitations:
    *   - Cannot add different breakpoints for different instances of the same module
    *   - Must use CMR API for all references
    */
  property("Should inject transaction logic") {

    val (ir, b) = Driver.elaborateAndReturn(() => new Buffer(4))

    val bps = BreakPoint("bpA", b.a1, (a: A, cmr: CMR) => {
      cmr(a.myreg) === 4.U
    }) //++ b.addBreakpoints

    val verilog = compile(ir, bps)
    println(verilog)
    assert(countModules(verilog) === 3)
  }

  /**
    * Limitations:
    *   - Cannot check paths through children instances
    *   - No cross module references
    */
  property("Should assert delays between signals") {

    class A(nRegs: Int) extends MultiIOModule {
      val in = IO(Input(UInt(3.W)))
      val out = IO(Output(UInt(3.W)))
      val regs = Reg(t=Vec(4, UInt(3.W)))
      out := regs.foldLeft(in){ (source, r) =>
        r := source
        r
      }
      // Annotate as part of RTL description
      annotate(AssertDelay(this, in, out, nRegs))
    }

    // Errors if not correct delay
    a [Exception] should be thrownBy {
      compile(new A(3))
    }

    // No error if correct delay
    val verilog = compile(new A(4))

    // Can add test-specific assertion
    val (ir, top) = Driver.elaborateAndReturn(() => new A(4))
    val badAssertion = AssertDelay(top, top.in, top.out, 3)
    a [Exception] should be thrownBy {
      compile(ir, Seq(badAssertion))
    }
  }

  property("Should track number of cycles in each state") {

    val (ir, topA) = Driver.elaborateAndReturn(() => new A)

    val aspects = ModuleAspect("histogram", topA, () => new Histogram, (a: A, histogram: Histogram) => {
      Map(
        a.clock -> histogram.clock,
        a.reset -> histogram.reset,
        a.reg -> histogram.in
      )
    })

    val verilog = compile(ir, aspects)
    println(verilog)
    assert(countModules(verilog) === 2)
  }

}

/* Histogram example for future use case
*/
class A extends MultiIOModule {
  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  val reg = RegInit(UInt(3.W), 0.U)
  reg := in
  out := reg
}

class Histogram extends MultiIOModule {
  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  val histMem = Mem(math.pow(2, in.getWidth).toInt, UInt(100.W))
  val readPort = histMem.read(in)
  histMem.write(in, readPort + 1.U)
  out := readPort
  dontTouch(out)
}
