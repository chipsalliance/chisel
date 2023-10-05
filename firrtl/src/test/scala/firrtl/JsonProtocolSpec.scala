// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import org.json4s._

import firrtl.annotations.{
  Annotation,
  HasSerializationHints,
  InvalidAnnotationJSONException,
  JsonProtocol,
  NoTargetAnnotation
}
import org.scalatest.flatspec.AnyFlatSpec

object JsonProtocolTestClasses {
  trait Parent

  case class ChildA(foo: Int) extends Parent
  case class ChildB(bar: String) extends Parent
  case class PolymorphicParameterAnnotation(param: Parent) extends NoTargetAnnotation
  case class PolymorphicParameterAnnotationWithTypeHints(param: Parent)
      extends NoTargetAnnotation
      with HasSerializationHints {
    def typeHints = Seq(param.getClass)
  }

  case class TypeParameterizedAnnotation[T](param: T) extends NoTargetAnnotation
  case class TypeParameterizedAnnotationWithTypeHints[T](param: T)
      extends NoTargetAnnotation
      with HasSerializationHints {
    def typeHints = Seq(param.getClass)
  }

  case class SimpleAnnotation(alpha: String) extends NoTargetAnnotation
}

import JsonProtocolTestClasses._

class JsonProtocolSpec extends AnyFlatSpec {
  def serializeAndDeserialize(anno: Annotation): Annotation = {
    val serializedAnno = JsonProtocol.serialize(Seq(anno))
    JsonProtocol.deserialize(serializedAnno).head
  }

  "Annotations with polymorphic parameters" should "not serialize and deserialize without type hints" in {
    val anno = PolymorphicParameterAnnotation(ChildA(1))
    assertThrows[InvalidAnnotationJSONException] {
      serializeAndDeserialize(anno)
    }
  }

  it should "serialize and deserialize with type hints" in {
    val anno = PolymorphicParameterAnnotationWithTypeHints(ChildA(1))
    val deserAnno = serializeAndDeserialize(anno)
    assert(anno == deserAnno)

    val anno2 = PolymorphicParameterAnnotationWithTypeHints(ChildB("Test"))
    val deserAnno2 = serializeAndDeserialize(anno2)
    assert(anno2 == deserAnno2)
  }

  "Annotations with non-primitive type parameters" should "not serialize and deserialize without type hints" in {
    val anno = TypeParameterizedAnnotation(ChildA(1))
    val deserAnno = serializeAndDeserialize(anno)
    assert(anno != deserAnno)
  }
  it should "serialize and deserialize with type hints" in {
    val anno = TypeParameterizedAnnotationWithTypeHints(ChildA(1))
    val deserAnno = serializeAndDeserialize(anno)
    assert(anno == deserAnno)
  }

  "JSON object order" should "not affect deserialization" in {
    val anno = SimpleAnnotation("hello")
    val serializedAnno = """[{
      "alpha": "hello",
      "class": "firrtlTests.JsonProtocolTestClasses$SimpleAnnotation"
    }]"""
    val deserAnno = JsonProtocol.deserialize(serializedAnno).head
    assert(anno == deserAnno)
  }
}
