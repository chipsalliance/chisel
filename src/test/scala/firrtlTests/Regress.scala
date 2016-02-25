
package firrtlTests

import org.scalatest._

import firrtl._
import java.io._
import scala.io.Source

class RocketRegressionSpec extends FlatSpec with Matchers {

  // This test is temporary until we move to simulation-based testing
  "CHIRRTL Rocket" should "match expected Verilog" in {
    val firrtlSource = Source.fromURL(getClass.getResource("/regress/rocket.fir"))
    val highCircuit = firrtl.Parser.parse("rocket.fir", firrtlSource.getLines)
    val verilogSW = new StringWriter()
    VerilogCompiler.run(highCircuit, verilogSW)

    val goldenVerilog = Source.fromURL(getClass.getResource("/regress/rocket-golden.v"))
    
    verilogSW.toString.split("\n") zip goldenVerilog.getLines.toSeq foreach {
      case (verilog, golden) => verilog shouldEqual golden
    }
  }
}
