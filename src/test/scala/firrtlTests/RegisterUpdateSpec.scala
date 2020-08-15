// See LICENSE for license details.

package firrtlTests

import firrtl._
import firrtl.ir._
import firrtl.transforms.FlattenRegUpdate
import firrtl.annotations.NoTargetAnnotation
import firrtl.stage.transforms.Compiler
import firrtl.options.Dependency
import firrtl.testutils._
import FirrtlCheckers._
import scala.util.matching.Regex

object RegisterUpdateSpec {
  case class CaptureStateAnno(value: CircuitState) extends NoTargetAnnotation
  // Capture the CircuitState between FlattenRegUpdate and VerilogEmitter
  //Emit captured state as FIRRTL for use in testing
  class CaptureCircuitState extends Transform with DependencyAPIMigration {
    override def prerequisites = Dependency[FlattenRegUpdate] :: Nil
    override def optionalPrerequisiteOf = Dependency[VerilogEmitter] :: Nil
    override def invalidates(a: Transform): Boolean = false
    def execute(state: CircuitState): CircuitState = {
      val emittedAnno = EmittedFirrtlCircuitAnnotation(
        EmittedFirrtlCircuit(state.circuit.main, state.circuit.serialize, ".fir")
      )
      val capturedState = state.copy(annotations = emittedAnno +: state.annotations)
      state.copy(annotations = CaptureStateAnno(capturedState) +: state.annotations)
    }
  }
}

class RegisterUpdateSpec extends FirrtlFlatSpec {
  import RegisterUpdateSpec._
  def compile(input: String): CircuitState = {
    val compiler = new Compiler(Seq(Dependency[CaptureCircuitState], Dependency[VerilogEmitter]))
    compiler.execute(CircuitState(parse(input), EmitCircuitAnnotation(classOf[VerilogEmitter]) :: Nil))
  }
  def compileBody(body: String) = {
    val str = """
                |circuit Test :
                |  module Test :
                |""".stripMargin + body.split("\n").mkString("    ", "\n    ", "")
    compile(str)
  }

  "Register update logic" should "not duplicate common subtrees" in {
    val result = compileBody(s"""
                                |input clock : Clock
                                |output io : { flip in : UInt<8>, flip a : UInt<1>, flip b : UInt<1>, flip c : UInt<1>, out : UInt<8>}
                                |reg r : UInt<8>, clock
                                |when io.a :
                                |  r <= io.in
                                |when io.b :
                                |  when io.c :
                                |    r <= UInt(2)
                                |io.out <= r""".stripMargin)
    // Checking intermediate state between FlattenRegUpdate and Verilog emission
    val fstate = result.annotations.collectFirst { case CaptureStateAnno(x) => x }.get
    fstate should containLine("""r <= mux(io_b, mux(io_c, UInt<8>("h2"), _GEN_0), _GEN_0)""")
    // Checking the Verilog
    val verilog = result.getEmittedCircuit.value
    result shouldNot containLine("r <= io_in;")
    verilog shouldNot include("if (io_a) begin")
    result should containLine("r <= _GEN_0;")
  }

  it should "not let duplicate subtrees on one register affect another" in {

    val result = compileBody(s"""
                                |input clock : Clock
                                |output io : { flip in : UInt<8>, flip a : UInt<1>, flip b : UInt<1>, flip c : UInt<1>, out : UInt<8>}

                                |reg r : UInt<8>, clock
                                |reg r2 : UInt<8>, clock
                                |when io.a :
                                |  r <= io.in
                                |  r2 <= io.in
                                |when io.b :
                                |  r2 <= UInt(3)
                                |  when io.c :
                                |    r <= UInt(2)
                                |io.out <= and(r, r2)""".stripMargin)
    // Checking intermediate state between FlattenRegUpdate and Verilog emission
    val fstate = result.annotations.collectFirst { case CaptureStateAnno(x) => x }.get
    fstate should containLine("""r <= mux(io_b, mux(io_c, UInt<8>("h2"), _GEN_0), _GEN_0)""")
    fstate should containLine("""r2 <= mux(io_b, UInt<8>("h3"), mux(io_a, io_in, r2))""")
    // Checking the Verilog
    val verilog = result.getEmittedCircuit.value
    result shouldNot containLine("r <= io_in;")
    result should containLine("r <= _GEN_0;")
    result should containLine("r2 <= io_in;")
    verilog should include("if (io_a) begin") // For r2
    // 1 time for r2, old versions would have 3 occurences
    Regex.quote("if (io_a) begin").r.findAllMatchIn(verilog).size should be(1)
  }

}
