// See LICENSE for license details.

package firrtlTests

import firrtl._
import org.scalatest.{Matchers, FreeSpec}

class ExecutionOptionManagerSpec extends FreeSpec with Matchers {
  "ExecutionOptionManager is a container for one more more ComposableOptions Block" - {
    "It has a default CommonOptionsBlock" in {
      val manager = new ExecutionOptionsManager("test")
      manager.commonOptions.targetDirName should be (".")
    }
    "But can override defaults like this" in {
      val manager = new ExecutionOptionsManager("test") { commonOptions = CommonOptions(topName = "dog") }
      manager.commonOptions shouldBe a [CommonOptions]
      manager.topName should be ("dog")
      manager.commonOptions.topName should be ("dog")
    }
    "The add method should put a new version of a given type the manager" in {
      val manager = new ExecutionOptionsManager("test") { commonOptions = CommonOptions(topName = "dog") }
      val initialCommon = manager.commonOptions
      initialCommon.topName should be ("dog")

      manager.commonOptions = CommonOptions(topName = "cat")

      val afterCommon = manager.commonOptions
      afterCommon.topName should be ("cat")
      initialCommon.topName should be ("dog")
    }
    "multiple composable blocks should be separable" in {
      val manager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
        commonOptions = CommonOptions(topName = "spoon")
        firrtlOptions = FirrtlExecutionOptions(inputFileNameOverride = "fork")
      }

      manager.firrtlOptions.inputFileNameOverride should be ("fork")
      manager.commonOptions.topName should be ("spoon")
    }
  }
}
