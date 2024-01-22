// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.panama.circt

class PanamaCIRCTOM private[chisel3] (circt: PanamaCIRCT, mlirModule: MlirModule) {
  def evaluator(): PanamaCIRCTOMEvaluator = new PanamaCIRCTOMEvaluator(circt, mlirModule)

  def newBasePathEmpty(): PanamaCIRCTOMEvaluatorValueBasePath =
    new PanamaCIRCTOMEvaluatorValueBasePath(circt, circt.omEvaluatorBasePathGetEmpty())
}

class PanamaCIRCTOMEvaluator private[chisel3] (circt: PanamaCIRCT, mlirModule: MlirModule) {
  val evaluator = circt.omEvaluatorNew(mlirModule)

  def instantiate(className: String, actualParams: Seq[PanamaCIRCTOMEvaluatorValue]): PanamaCIRCTOMObject = {
    val params = actualParams.map(_.asInstanceOf[PanamaCIRCTOMEvaluatorValue].value)

    val value = circt.omEvaluatorInstantiate(evaluator, className, params)
    assert(!circt.omEvaluatorObjectIsNull(value))
    new PanamaCIRCTOMObject(circt, value)
  }
}

abstract class PanamaCIRCTOMEvaluatorValue {
  val value: OMEvaluatorValue
}
object PanamaCIRCTOMEvaluatorValue {
  def newValue(circt: PanamaCIRCT, value: OMEvaluatorValue): PanamaCIRCTOMEvaluatorValue = {
    if (circt.omEvaluatorValueIsAObject(value)) {
      throw new Exception("TODO")
    } else if (circt.omEvaluatorValueIsAPrimitive(value)) {
      new PanamaCIRCTOMEvaluatorValuePrimitive(circt, value)
    } else if (circt.omEvaluatorValueIsAList(value)) {
      new PanamaCIRCTOMEvaluatorValueList(circt, value)
    } else if (circt.omEvaluatorValueIsATuple(value)) {
      new PanamaCIRCTOMEvaluatorValueTuple(circt, value)
    } else if (circt.omEvaluatorValueIsAMap(value)) {
      new PanamaCIRCTOMEvaluatorValueMap(circt, value)
    } else if (circt.omEvaluatorValueIsABasePath(value)) {
      new PanamaCIRCTOMEvaluatorValueBasePath(circt, value)
    } else if (circt.omEvaluatorValueIsAPath(value)) {
      new PanamaCIRCTOMEvaluatorValuePath(circt, value)
    } else {
      throw new Exception("unknown OMEvaluatorValue type")
    }
  }
}

class PanamaCIRCTOMEvaluatorValueList private[chisel3] (circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  val numElements: Long = circt.omEvaluatorListGetNumElements(value)
  def getElement(index: Long) =
    PanamaCIRCTOMEvaluatorValue.newValue(circt, circt.omEvaluatorListGetElement(value, index))
}

class PanamaCIRCTOMEvaluatorValueTuple private[chisel3] (circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  val numElements: Long = circt.omEvaluatorTupleGetNumElements(value)
  def getElement(index: Long) =
    PanamaCIRCTOMEvaluatorValue.newValue(circt, circt.omEvaluatorTupleGetElement(value, index))
}

class PanamaCIRCTOMEvaluatorValueMap private[chisel3] (circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  val tpe:  MlirType = circt.omEvaluatorMapGetType(value)
  val keys: MlirAttribute = circt.omEvaluatorMapGetKeys(value)
  def getElement(attr: MlirAttribute) =
    PanamaCIRCTOMEvaluatorValue.newValue(circt, circt.omEvaluatorMapGetElement(value, attr))
}

class PanamaCIRCTOMEvaluatorValueBasePath private[chisel3] (circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {}

class PanamaCIRCTOMEvaluatorValuePath private[chisel3] (circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  def asString(): String = circt.omEvaluatorPathGetAsString(value)
}

class PanamaCIRCTOMEvaluatorValuePrimitive private[chisel3] (circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  val primitive: MlirAttribute = circt.omEvaluatorValueGetPrimitive(value)
}

class PanamaCIRCTOMObject private[chisel3] (circt: PanamaCIRCT, value: OMEvaluatorValue) {
  def field(name: String): PanamaCIRCTOMEvaluatorValue =
    PanamaCIRCTOMEvaluatorValue.newValue(circt, circt.omEvaluatorObjectGetField(value, name))
}
