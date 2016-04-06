
package firrtlTests

import org.scalatest._

// Miscellaneous Feature Checks
class FeatureSpec extends FirrtlPropSpec {

  property("Nested SubAcceses should be supported!") {
    runFirrtlTest("NestedSubAccessTester", "/features")
  }
}

