// See LICENSE for license details.

package firrtlTests

class SimpleExtModuleExecutionTest extends ExecutionTest("SimpleExtModuleTester", "/blackboxes",
                                                         Seq("SimpleExtModule"))
class MultiExtModuleExecutionTest extends ExecutionTest("MultiExtModuleTester", "/blackboxes",
                                                        Seq("SimpleExtModule", "AdderExtModule"))
class RenamedExtModuleExecutionTest extends ExecutionTest("RenamedExtModuleTester", "/blackboxes",
                                                          Seq("SimpleExtModule"))
class ParameterizedExtModuleExecutionTest extends ExecutionTest(
    "ParameterizedExtModuleTester", "/blackboxes", Seq("ParameterizedExtModule"))

