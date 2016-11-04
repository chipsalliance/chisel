// See LICENSE for license details.

package firrtlTests

import org.scalatest._
import org.scalatest.prop._

class GCDExecutionTest extends ExecutionTest("GCDTester", "/integration")
class RightShiftExecutionTest extends ExecutionTest("RightShiftTester", "/integration")
class MemExecutionTest extends ExecutionTest("MemTester", "/integration")

class RocketCompilationTest extends CompilationTest("rocket", "/regress")
class RocketFirrtlCompilationTest extends CompilationTest("rocket-firrtl", "/regress")
class BOOMRobCompilationTest extends CompilationTest("Rob", "/regress")

