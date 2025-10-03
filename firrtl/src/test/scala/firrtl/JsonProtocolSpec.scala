// SPDX-License-Identifier: Apache-2.0

package firrtlTests

import scala.util.Failure

import org.json4s._

import firrtl.annotations._
import firrtl.transforms.DontTouchAnnotation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.Inside._
import org.scalatest.matchers.should._

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

  case class AnnotationWithOverride(value: String) extends NoTargetAnnotation with OverrideSerializationClass {
    def serializationClassOverride = value
  }

  // Test case for OverrideSerializationClass on nested types
  case class NestedTypeWithOverride(name: String) extends OverrideSerializationClass {
    def serializationClassOverride = "custom.nested.type"
  }

  case class AnnotationWithNestedOverride(nested: NestedTypeWithOverride)
      extends NoTargetAnnotation
      with HasSerializationOverrides {
    def typeOverrides = Seq(nested.getClass -> nested.serializationClassOverride)
  }
}

import JsonProtocolTestClasses._

class JsonProtocolSpec extends AnyFlatSpec with Matchers {
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

  "JsonProtocol" should "support serializing directly to a Java Writer" in {
    val anno = SimpleAnnotation("hello")
    class NaiveWriter extends java.io.Writer {
      private var contents: String = ""
      def value:            String = contents
      def close():          Unit = contents = ""
      def flush():          Unit = contents = ""
      def write(cbuff: Array[Char], off: Int, len: Int): Unit = {
        for (i <- off until off + len) {
          contents += cbuff(i)
        }
      }
    }
    val w = new NaiveWriter
    JsonProtocol.serializeTry(Seq(anno), w)
    val ser1 = w.value
    val ser2 = JsonProtocol.serialize(Seq(anno))
    assert(ser1 == ser2)
  }

  "Trying to serialize annotations that cannot be serialized" should "tell you why" in {
    case class MyAnno(x: Int) extends NoTargetAnnotation
    inside(JsonProtocol.serializeTry(MyAnno(3) :: Nil)) { case Failure(e: UnserializableAnnotationException) =>
      e.getMessage should include("MyAnno")
      // From json4s Exception
      e.getMessage should include("Classes defined in method bodies are not supported")
    }
  }

  "JsonProtocol.serializeRecover" should "emit even annotations that cannot be serialized" in {
    case class MyAnno(x: Int) extends NoTargetAnnotation
    val target = ModuleTarget("Foo").ref("x")
    val annos = MyAnno(3) :: DontTouchAnnotation(target) :: Nil
    val res = JsonProtocol.serializeRecover(annos)
    res should include(""""class":"firrtl.annotations.UnserializeableAnnotation",""")
    res should include(""""error":"Classes defined in method bodies are not supported.",""")
    res should include(""""content":"MyAnno(3)"""")
  }

  "OverrideSerializationClass" should "allow annotations to change the 'class' in JSON" in {
    val anno = AnnotationWithOverride("a.b.c.d.Foo")
    val res = JsonProtocol.serialize(Seq(anno))
    res should include(""""class":"a.b.c.d.Foo"""")
  }

  it should "work for the same class if it has the same override" in {
    val annos = AnnotationWithOverride("foo") :: AnnotationWithOverride("foo") :: Nil
    JsonProtocol.serialize(annos)
  }

  it should "error if inconsistent overrides are used" in {
    val annos = AnnotationWithOverride("foo") :: AnnotationWithOverride("bar") :: Nil
    val e = the[Exception] thrownBy JsonProtocol.serialize(annos)
    e.getMessage should include("multiple serialization class overrides: foo, bar")
  }

  it should "work on nested types inside annotations with HasSerializationOverrides" in {
    val nested = NestedTypeWithOverride("test")
    val anno = AnnotationWithNestedOverride(nested)
    val res = JsonProtocol.serialize(Seq(anno))
    // Verify that the nested type uses the overridden class name
    res should include(""""class":"custom.nested.type"""")
    // Also verify that the annotation itself uses its normal class name
    res should include(""""class":"firrtlTests.JsonProtocolTestClasses$AnnotationWithNestedOverride"""")
  }
}
