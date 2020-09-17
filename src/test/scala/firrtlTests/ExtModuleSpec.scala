// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.testutils._

class SimpleExtModuleExecutionTest extends ExecutionTest("SimpleExtModuleTester", "/blackboxes", Seq("SimpleExtModule"))
class MultiExtModuleExecutionTest
    extends ExecutionTest("MultiExtModuleTester", "/blackboxes", Seq("SimpleExtModule", "AdderExtModule"))
class RenamedExtModuleExecutionTest
    extends ExecutionTest("RenamedExtModuleTester", "/blackboxes", Seq("SimpleExtModule"))
class ParameterizedExtModuleExecutionTest
    extends ExecutionTest("ParameterizedExtModuleTester", "/blackboxes", Seq("ParameterizedExtModule"))

class LargeParamExecutionTest extends ExecutionTest("LargeParamTester", "/blackboxes", Seq("LargeParam"))
