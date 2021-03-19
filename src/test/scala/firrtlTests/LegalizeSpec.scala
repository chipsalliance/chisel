// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl.testutils.{ExecutionTest, ExecutionTestNoOpt}

class LegalizeExecutionTest extends ExecutionTest("Legalize", "/passes/Legalize")
// Legalize also needs to work when optimizations are turned off
class LegalizeExecutionTestNoOpt extends ExecutionTestNoOpt("Legalize", "/passes/Legalize")
