// SPDX-License-Identifier: Apache-2.0

package chisel3.panamaom

import chisel3.panamalib._

class PanamaCIRCTOM private[chisel3] (circt: PanamaCIRCT, mlirModule: MlirModule) {
  def evaluator(): PanamaCIRCTOMEvaluator = new PanamaCIRCTOMEvaluator(circt, mlirModule)

  def newBasePathEmpty(): PanamaCIRCTOMEvaluatorValueBasePath =
    new PanamaCIRCTOMEvaluatorValueBasePath(circt, circt.omEvaluatorBasePathGetEmpty())
}

class PanamaCIRCTOMEvaluator private[chisel3] (circt: PanamaCIRCT, mlirModule: MlirModule) {
  val evaluator = circt.omEvaluatorNew(mlirModule)

  def instantiate(
    className:    String,
    actualParams: Seq[PanamaCIRCTOMEvaluatorValue]
  ): PanamaCIRCTOMEvaluatorValueObject = {
    val params = actualParams.map(_.asInstanceOf[PanamaCIRCTOMEvaluatorValue].value)

    val value = circt.omEvaluatorInstantiate(evaluator, className, params)
    assert(!circt.omEvaluatorObjectIsNull(value))
    new PanamaCIRCTOMEvaluatorValueObject(circt, value)
  }
}

abstract class PanamaCIRCTOMEvaluatorValue {
  val circt: PanamaCIRCT
  val value: OMEvaluatorValue

  // Incomplete. currently for debugging purposes only
  override def toString: String = {
    this match {
      case v: PanamaCIRCTOMEvaluatorValuePath      => s"path{${v.toString}}"
      case v: PanamaCIRCTOMEvaluatorValueList      => s"[ ${v.elements.map(_.toString).mkString(", ")} ]"
      case v: PanamaCIRCTOMEvaluatorValuePrimitive => s"prim{${v.toString}}"
      case v: PanamaCIRCTOMEvaluatorValueObject    =>
        val subfields = v.fieldNames
          .map(name => (name, v.field(name)))
          .map { case (name, value) => s".$name => { ${value.toString} }" }
          .mkString(", ")
        s"obj{$subfields}"
    }
  }
}
object PanamaCIRCTOMEvaluatorValue {
  def newValue(circt: PanamaCIRCT, value: OMEvaluatorValue): PanamaCIRCTOMEvaluatorValue = {
    if (circt.omEvaluatorValueIsAObject(value)) {
      new PanamaCIRCTOMEvaluatorValueObject(circt, value)
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

class PanamaCIRCTOMEvaluatorValueList private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  val numElements: Long = circt.omEvaluatorListGetNumElements(value)
  def getElement(index: Long) =
    PanamaCIRCTOMEvaluatorValue.newValue(circt, circt.omEvaluatorListGetElement(value, index))
  def elements(): Seq[PanamaCIRCTOMEvaluatorValue] =
    (0.toLong until numElements).map { i =>
      getElement(i)
    }
}

class PanamaCIRCTOMEvaluatorValueTuple private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  val numElements: Long = circt.omEvaluatorTupleGetNumElements(value)
  def getElement(index: Long) =
    PanamaCIRCTOMEvaluatorValue.newValue(circt, circt.omEvaluatorTupleGetElement(value, index))
}

class PanamaCIRCTOMEvaluatorValueMap private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  val tpe:  MlirType = circt.omEvaluatorMapGetType(value)
  val keys: MlirAttribute = circt.omEvaluatorMapGetKeys(value)
  def getElement(attr: MlirAttribute) =
    PanamaCIRCTOMEvaluatorValue.newValue(circt, circt.omEvaluatorMapGetElement(value, attr))
}

class PanamaCIRCTOMEvaluatorValueBasePath private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {}

class PanamaCIRCTOMEvaluatorValuePath private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  override def toString: String = circt.omEvaluatorPathGetAsString(value)
}

class PanamaCIRCTOMEvaluatorValuePrimitive private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  val primitive: MlirAttribute = circt.omEvaluatorValueGetPrimitive(value)

  // Incomplete. currently for debugging purposes only
  override def toString: String = {
    if (circt.omAttrIsAIntegerAttr(primitive)) {
      val mlirInteger = circt.omIntegerAttrGetInt(primitive)
      val integer = circt.mlirIntegerAttrGetValueSInt(mlirInteger)
      s"omInteger{$integer}"
    } else if (circt.mlirAttributeIsAString(primitive)) {
      val mlirString = circt.mlirStringAttrGetValue(primitive)
      s"mlirString{\"$mlirString\"}"
    } else {
      circt.mlirAttributeDump(primitive)
      throw new Exception("unhandled primitive type dumped")
    }
  }
}

class PanamaCIRCTOMEvaluatorValueObject private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  def field(name: String): PanamaCIRCTOMEvaluatorValue =
    PanamaCIRCTOMEvaluatorValue.newValue(circt, circt.omEvaluatorObjectGetField(value, name))

  def fieldNames(): Seq[String] = {
    val names = circt.omEvaluatorObjectGetFieldNames(value)
    val numNames = circt.mlirArrayAttrGetNumElements(names)
    Seq.tabulate(numNames.toInt)(identity).map { i =>
      val name = circt.mlirArrayAttrGetElement(names, i)
      circt.mlirStringAttrGetValue(name)
    }
  }

  def foreachField(f: (String, PanamaCIRCTOMEvaluatorValue) => Unit): Unit = {
    fieldNames.map { name =>
      f(name, field(name))
    }
  }
}
