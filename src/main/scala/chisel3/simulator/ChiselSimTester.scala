package chisel3
package simulator

import org.scalatest.TestSuite
import org.scalatest.Assertions
import org.scalatest.TestSuiteMixin
import org.scalatest.Outcome

import scala.util.DynamicVariable

trait ScalatestChiselSimTester extends ChiselSimAPI with Assertions with TestSuiteMixin { this: TestSuite =>

  override def testName: Option[String] = scalaTestContext.value.map(_.name)

  protected var scalaTestContext = new DynamicVariable[Option[NoArgTest]](None)

  abstract override def withFixture(test: NoArgTest): Outcome = {
    require(scalaTestContext.value.isEmpty, "scalaTestContext already set")
    scalaTestContext.withValue(Some(test)) {
      super.withFixture(test)
    }
  }

  // TODO: ability to override settings from command line
  protected def testConfig = scalaTestContext.value.get.configMap
}

trait ChiselSimTester extends ThreadedChiselSimAPI with ScalatestChiselSimTester { this: TestSuite => }
