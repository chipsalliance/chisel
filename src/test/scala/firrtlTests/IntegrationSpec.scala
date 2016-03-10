
package firrtlTests

import org.scalatest._
import org.scalatest.prop._

class IntegrationSpec extends FirrtlPropSpec {

  case class Test(name: String, dir: String)

  val runTests = Seq(Test("GCDTester", "/integration"),
                     Test("RightShiftTester", "/integration"))
      

  runTests foreach { test =>
    property(s"${test.name} should execute correctly") {
      runFirrtlTest(test.name, test.dir)
    }
  }

  val compileTests = Seq(Test("rocket", "/regress"), Test("rocket-firrtl", "/regress"))

  compileTests foreach { test =>
    property(s"${test.name} should compile to Verilog") {
      compileFirrtlTest(test.name, test.dir)
    }
  }
}
