// See LICENSE.txt for details


package ciclTests


import chiselTests.ChiselPropSpec
import chisel3._
import chisel3.experimental.MultiIOModule
import chisel3.libs.aspect.{CrossModule, Snippet}
import chisel3.libs.transaction.TransactionEvent
import chisel3.libs.diagnostic.{DelayCounter, DelayCounterAnnotation, Histogram}

class Buffer(delay: Int) extends MultiIOModule {
  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  val regs = Seq.fill(delay)(Reg(UInt(3.W)))
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
    *   - Cannot add different transactions for different instances of the same module
    *   - Must use CMR API for all references
    *   - Cannot do instance annotations
    */
  property("Should inject transaction logic") {

    val (ir, b) = Driver.elaborateAndReturn(() => new Buffer(1))

    val xactions = TransactionEvent("is0123",  b, new Snippet[Buffer, Bool] {
      override def snip(top: Buffer)  = {
        top.regs.map(_.i).zipWithIndex.map {
          case (reg, index) => reg === index.U
        }.reduce(_ & _)
      }
    })
    xactions.foreach(println)

    val verilog = compile(ir, xactions)
    println(verilog)
    assert(countModules(verilog) === 2)
  }

  property("Should track number of cycles in each state") {

    val (ir, topA) = Driver.elaborateAndReturn(() => new Buffer(1))

    val aspects = Histogram("histReg0", topA, topA.regs(0), 100)
    println(aspects)

    val verilog = compile(ir, aspects)
    println(verilog)
    assert(countModules(verilog) === 2)
  }

  /**
    * Limitations:
    *   - Cannot check paths through children instances
    *   - No cross module references
    */
  property("Should count delays between signals") {

    val (ir, b) = Driver.elaborateAndReturn(() => new Buffer(4))

    val countDelays = CrossModule.withRoot(b) {
      DelayCounter(b, b.in, b.out)
    }

    val (verilog, state) = compileAndReturn(ir, countDelays)

    println(verilog)
    assert(countModules(verilog) === 1)

    val diagnostics = state.annotations.collect{case DelayCounterAnnotation(_, _, _, Some(delays)) => delays}
    assert(diagnostics.size == 1 && diagnostics.head == Set(4))
  }

}



/**
  * Limitations:
  *   - Cannot check paths through children instances
  *   - No cross module references
  */
  /*
property("Should count delays between signals") {

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
*/
