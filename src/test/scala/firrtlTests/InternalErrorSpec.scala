// See LICENSE for license details.

package firrtlTests

import java.io.File

import firrtl._
import firrtl.Utils.getThrowable
import firrtl.util.BackendCompilationUtilities
import org.scalatest.{FreeSpec, Matchers}


class InternalErrorSpec extends FreeSpec with Matchers with BackendCompilationUtilities {
  "Unexpected exceptions" - {
    val input =
      """
        |circuit Dummy :
        |  module Dummy :
        |    input clock : Clock
        |    input x : UInt<1>
        |    output y : UInt<1>
        |    output io : { flip in : UInt<16>, out : UInt<16> }
        |    y <= shr(x, UInt(1)); this should generate an exception in PrimOps.scala:127.
        |      """.stripMargin

    var exception: Exception = null
    "should throw a FIRRTLException" in {
      val manager = new ExecutionOptionsManager("test") with HasFirrtlOptions {
        commonOptions = CommonOptions(topName = "Dummy")
        firrtlOptions = FirrtlExecutionOptions(firrtlSource = Some(input), compilerName = "low")
      }
      exception = intercept[FIRRTLException] {
        firrtl.Driver.execute(manager)
      }
    }

    "should contain the expected string" in {
      assert(exception.getMessage.contains("Internal Error! Please file an issue"))
    }

    "should contain the name of the file originating the exception in the stack trace" in {
      val first = true
      assert(getThrowable(Some(exception), first).getStackTrace exists (_.getFileName.contains("PrimOps.scala")))
    }
  }
}
