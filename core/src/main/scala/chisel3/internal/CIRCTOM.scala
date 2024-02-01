// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

@deprecated("There no CIRCTOM anymore, use circtpanamaom directly", "Chisel 6.0")
abstract class CIRCTOM {
  def evaluator(): CIRCTOMEvaluator

  def newBasePathEmpty(): CIRCTOMEvaluatorValue
}

@deprecated("There no CIRCTOMEvaluator anymore, use circtpanamaom directly", "Chisel 6.0")
abstract class CIRCTOMEvaluator {
  def instantiate(name: String, actualParams: Seq[CIRCTOMEvaluatorValue]): CIRCTOMObject
}

@deprecated("There no CIRCTOMEvaluatorValue anymore, use circtpanamaom directly", "Chisel 6.0")
abstract class CIRCTOMEvaluatorValue {}

@deprecated("There no CIRCTOMObject anymore, use circtpanamaom directly", "Chisel 6.0")
abstract class CIRCTOMObject {
  def field(name: String): CIRCTOMEvaluatorValue
}
