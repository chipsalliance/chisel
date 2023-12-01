// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

abstract class CIRCTOM {
  def evaluator(): CIRCTOMEvaluator

  def newBasePathEmpty(): CIRCTOMEvaluatorValue
}

abstract class CIRCTOMEvaluator {
  def instantiate(name: String, actualParams: Seq[CIRCTOMEvaluatorValue]): CIRCTOMObject
}

abstract class CIRCTOMEvaluatorValue {}

abstract class CIRCTOMObject {
  def field(name: String): CIRCTOMEvaluatorValue
}
