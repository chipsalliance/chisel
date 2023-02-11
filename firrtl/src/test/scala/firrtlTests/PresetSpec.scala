// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import firrtl._
import firrtl.annotations._
import firrtl.testutils._
import firrtl.testutils.FirrtlCheckers._
import logger.{LogLevel, LogLevelAnnotation, Logger}

class PresetExecutionTest
    extends ExecutionTest(
      "PresetTester",
      "/features",
      annotations = Seq(new PresetAnnotation(CircuitTarget("PresetTester").module("PresetTester").ref("preset")))
    )
