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
  ): Option[PanamaCIRCTOMEvaluatorValueObject] = {
    val params = actualParams.map(_.asInstanceOf[PanamaCIRCTOMEvaluatorValue].value)

    val value = circt.omEvaluatorInstantiate(evaluator, className, params)
    if (!circt.omEvaluatorObjectIsNull(value)) {
      Some(new PanamaCIRCTOMEvaluatorValueObject(circt, value))
    } else {
      None
    }
  }
}

abstract class PanamaCIRCTOMEvaluatorValue {
  val circt: PanamaCIRCT
  val value: OMEvaluatorValue
  def objOpt: Option[PanamaCIRCTOMEvaluatorValueObject] =
    this match {
      case valueObject: PanamaCIRCTOMEvaluatorValueObject => Some(valueObject)
      case _ => None
    }
  def obj: PanamaCIRCTOMEvaluatorValueObject = objOpt.get

  def basePathOpt: Option[PanamaCIRCTOMEvaluatorValueBasePath] =
    this match {
      case path: PanamaCIRCTOMEvaluatorValueBasePath => Some(path)
      case _ => None
    }
  def basePath: PanamaCIRCTOMEvaluatorValueBasePath = basePathOpt.get

  def listOpt: Option[PanamaCIRCTOMEvaluatorValueList] =
    this match {
      case list: PanamaCIRCTOMEvaluatorValueList => Some(list)
      case _ => None
    }
  def list: PanamaCIRCTOMEvaluatorValueList = listOpt.get

  def pathOpt: Option[PanamaCIRCTOMEvaluatorValuePath] =
    this match {
      case path: PanamaCIRCTOMEvaluatorValuePath => Some(path)
      case _ => None
    }
  def path: PanamaCIRCTOMEvaluatorValuePath = pathOpt.get

  def intOpt: Option[PanamaCIRCTOMEvaluatorValuePrimitiveInteger] =
    this match {
      case integer: PanamaCIRCTOMEvaluatorValuePrimitiveInteger => Some(integer)
      case _ => None
    }
  def int: PanamaCIRCTOMEvaluatorValuePrimitiveInteger = intOpt.get

  def floatOpt: Option[PanamaCIRCTOMEvaluatorValuePrimitiveFloat] =
    this match {
      case float: PanamaCIRCTOMEvaluatorValuePrimitiveFloat => Some(float)
      case _ => None
    }
  def float: PanamaCIRCTOMEvaluatorValuePrimitiveFloat = floatOpt.get

  def stringOpt: Option[PanamaCIRCTOMEvaluatorValuePrimitiveString] =
    this match {
      case string: PanamaCIRCTOMEvaluatorValuePrimitiveString => Some(string)
      case _ => None
    }
  def string: PanamaCIRCTOMEvaluatorValuePrimitiveString = stringOpt.get

  def boolOpt: Option[PanamaCIRCTOMEvaluatorValuePrimitiveBool] =
    this match {
      case bool: PanamaCIRCTOMEvaluatorValuePrimitiveBool => Some(bool)
      case _ => None
    }
  def bool: PanamaCIRCTOMEvaluatorValuePrimitiveBool = boolOpt.get

  // Incomplete. currently for debugging purposes only
  override def toString: String = {
    this match {
      case v: PanamaCIRCTOMEvaluatorValuePath      => s"path{${v.toString}}"
      case v: PanamaCIRCTOMEvaluatorValueList      => s"[ ${v.elements.map(_.toString).mkString(", ")} ]"
      case v: PanamaCIRCTOMEvaluatorValuePrimitive => s"prim{${v.toString}}"
      case v: PanamaCIRCTOMEvaluatorValueObject =>
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
    } else if (circt.omEvaluatorValueIsAList(value)) {
      new PanamaCIRCTOMEvaluatorValueList(circt, value)
    } else if (circt.omEvaluatorValueIsABasePath(value)) {
      new PanamaCIRCTOMEvaluatorValueBasePath(circt, value)
    } else if (circt.omEvaluatorValueIsAPath(value)) {
      new PanamaCIRCTOMEvaluatorValuePath(circt, value)
    } else if (circt.omEvaluatorValueIsAReference(value)) {
      newValue(circt, circt.omEvaluatorValueGetReferenceValue(value))
    } else if (circt.omEvaluatorValueIsAPrimitive(value)) {
      PanamaCIRCTOMEvaluatorValuePrimitive.newPrimitive(circt, value)
    } else if (circt.omEvaluatorValueIsNull(value)) {
      throw new Exception("unable to get field")
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

class PanamaCIRCTOMEvaluatorValueBasePath private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {}

class PanamaCIRCTOMEvaluatorValuePath private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  override def toString: String = circt.omEvaluatorPathGetAsString(value)
}

object PanamaCIRCTOMEvaluatorValuePrimitive {
  def newPrimitive(circt: PanamaCIRCT, value: OMEvaluatorValue): PanamaCIRCTOMEvaluatorValuePrimitive = {
    val primitive: MlirAttribute = circt.omEvaluatorValueGetPrimitive(value)

    if (circt.omAttrIsAIntegerAttr(primitive)) {
      new PanamaCIRCTOMEvaluatorValuePrimitiveInteger(circt, value, circt.omIntegerAttrGetInt(primitive))
    } else if (circt.mlirAttributeIsAFloat(primitive)) {
      new PanamaCIRCTOMEvaluatorValuePrimitiveFloat(circt, value, primitive)
    } else if (circt.mlirAttributeIsAString(primitive)) {
      new PanamaCIRCTOMEvaluatorValuePrimitiveString(circt, value, primitive)
    } else if (circt.mlirAttributeIsABool(primitive)) {
      new PanamaCIRCTOMEvaluatorValuePrimitiveBool(circt, value, primitive)
    } else {
      circt.mlirAttributeDump(primitive)
      throw new Exception("unknown OMEvaluatorValuePrimitive attribute, dumped")
    }
  }
}
abstract class PanamaCIRCTOMEvaluatorValuePrimitive extends PanamaCIRCTOMEvaluatorValue {
  override def toString: String = this match {
    case v: PanamaCIRCTOMEvaluatorValuePrimitiveInteger => v.toString
    case v: PanamaCIRCTOMEvaluatorValuePrimitiveFloat   => v.toString
    case v: PanamaCIRCTOMEvaluatorValuePrimitiveString  => v.toString
    case v: PanamaCIRCTOMEvaluatorValuePrimitiveBool    => v.toString
  }
}

class PanamaCIRCTOMEvaluatorValuePrimitiveInteger private[chisel3] (
  val circt:     PanamaCIRCT,
  val value:     OMEvaluatorValue,
  val primitive: MlirAttribute
) extends PanamaCIRCTOMEvaluatorValuePrimitive {
  val integer:           Long = circt.mlirIntegerAttrGetValueSInt(primitive)
  override def toString: String = integer.toString
}

class PanamaCIRCTOMEvaluatorValuePrimitiveFloat private[chisel3] (
  val circt:     PanamaCIRCT,
  val value:     OMEvaluatorValue,
  val primitive: MlirAttribute
) extends PanamaCIRCTOMEvaluatorValuePrimitive {
  val double:            Double = circt.mlirFloatAttrGetValueDouble(primitive)
  override def toString: String = double.toString
}

class PanamaCIRCTOMEvaluatorValuePrimitiveString private[chisel3] (
  val circt:     PanamaCIRCT,
  val value:     OMEvaluatorValue,
  val primitive: MlirAttribute
) extends PanamaCIRCTOMEvaluatorValuePrimitive {
  override def toString: String = circt.mlirStringAttrGetValue(primitive)
}

class PanamaCIRCTOMEvaluatorValuePrimitiveBool private[chisel3] (
  val circt:     PanamaCIRCT,
  val value:     OMEvaluatorValue,
  val primitive: MlirAttribute
) extends PanamaCIRCTOMEvaluatorValuePrimitive {
  val boolean:           Boolean = circt.mlirBoolAttrGetValue(primitive)
  override def toString: String = boolean.toString
}

class PanamaCIRCTOMEvaluatorValueObject private[chisel3] (val circt: PanamaCIRCT, val value: OMEvaluatorValue)
    extends PanamaCIRCTOMEvaluatorValue {
  def apply(name: String): PanamaCIRCTOMEvaluatorValue = field(name)
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
