// See LICENSE for license details.

package firrtlTests

import firrtl._
import org.scalatest.{Matchers, FreeSpec}

class ExecutionOptionsManagerSpec extends FreeSpec with Matchers {
  "ExecutionOptionsManager is a container for one more more ComposableOptions Block" - {
    "It has a default CommonOptionsBlock" in {
      val manager = new ExecutionOptionsManager("test")
      manager.topName should be ("")
      manager.targetDirName should be (".")
      manager.commonOptions.topName should be ("")
      manager.commonOptions.targetDirName should be (".")
    }
    "But can override defaults like this" in {
      val manager = new ExecutionOptionsManager("test") { commonOptions = CommonOptions(topName = "dog", targetDirName = "a/b/c") }
      manager.commonOptions shouldBe a [CommonOptions]
      manager.topName should be ("dog")
      manager.targetDirName should be ("a/b/c")
      manager.commonOptions.topName should be ("dog")
      manager.commonOptions.targetDirName should be ("a/b/c")
    }
    "The add method should put a new version of a given type the manager" in {
      val manager = new ExecutionOptionsManager("test") { commonOptions = CommonOptions(topName = "dog", targetDirName = "a/b/c") }
      val initialCommon = manager.commonOptions
      initialCommon.topName should be ("dog")
      initialCommon.targetDirName should be ("a/b/c")

      manager.commonOptions = CommonOptions(topName = "cat", targetDirName = "d/e/f")

      val afterCommon = manager.commonOptions
      afterCommon.topName should be ("cat")
      afterCommon.targetDirName should be ("d/e/f")
      initialCommon.topName should be ("dog")
      initialCommon.targetDirName should be ("a/b/c")
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
