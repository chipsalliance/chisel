// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.util.Decoupled
import org.scalatest.exceptions.TestFailedException
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

import scala.language.reflectiveCalls

/* Rich and complicated bundle examples
 */
trait BpipSuperTraitWithField {
  val bpipSuperTraitGood = SInt(17.W)
  def bpipSuperTraitBad = SInt(22.W)
}

trait BpipTraitWithField extends BpipSuperTraitWithField {
  val bpipTraitGood = SInt(17.W)
  def bpipTraitBad = SInt(22.W)
}

class BpipOneField extends Bundle with BpipTraitWithField {
  val bpipOneFieldOne = SInt(8.W)
  val bpipOneFieldTwo = SInt(8.W)
}

class BpipTwoField extends BpipOneField {
  val bpipTwoFieldOne = SInt(8.W)
  val bpipTwoFieldTwo = Vec(4, UInt(12.W))
  val myInt = 7
  val baz = Decoupled(UInt(16.W))
}

class BpipDecoupled extends BpipOneField {
  val bpipDecoupledSInt = SInt(8.W)
  val bpipDecoupledVec = Vec(4, UInt(12.W))
  val bpipDecoupledDecoupled = Decoupled(UInt(16.W))
}

class HasDecoupledBundleByInheritance extends Module {
  val out1 = IO(Output(new BpipDecoupled))
  assertElementsMatchExpected(out1)(
    "bpipDecoupledDecoupled" -> _.bpipDecoupledDecoupled,
    "bpipDecoupledVec" -> _.bpipDecoupledVec,
    "bpipDecoupledSInt" -> _.bpipDecoupledSInt,
    "bpipOneFieldTwo" -> _.bpipOneFieldTwo,
    "bpipOneFieldOne" -> _.bpipOneFieldOne,
    "bpipTraitGood" -> _.bpipTraitGood,
    "bpipSuperTraitGood" -> _.bpipSuperTraitGood
  )
}

/* plugin should not affect the seq detection */
class DebugProblem3 extends Module {
  val out1 = IO(Output(new BpipTwoField))
  assertElementsMatchExpected(out1)(
    "baz" -> _.baz,
    "bpipTwoFieldTwo" -> _.bpipTwoFieldTwo,
    "bpipTwoFieldOne" -> _.bpipTwoFieldOne,
    "bpipOneFieldTwo" -> _.bpipOneFieldTwo,
    "bpipOneFieldOne" -> _.bpipOneFieldOne,
    "bpipTraitGood" -> _.bpipTraitGood,
    "bpipSuperTraitGood" -> _.bpipSuperTraitGood
  )
}

class BpipBadSeqBundle extends Bundle {
  val bpipBadSeqBundleGood = UInt(999.W)
  val bpipBadSeqBundleBad = Seq(UInt(16.W), UInt(8.W), UInt(4.W))
}

class HasBadSeqBundle extends Module {
  val out1 = IO(Output(new BpipBadSeqBundle))
}

class BpipBadSeqBundleWithIgnore extends Bundle with IgnoreSeqInBundle {
  val goodFieldWithIgnore = UInt(999.W)
  val badSeqFieldWithIgnore = Seq(UInt(16.W), UInt(8.W), UInt(4.W))
}

class UsesIgnoreSeqInBundle extends Module {
  val out1 = IO(Output(new BpipBadSeqBundleWithIgnore))
}

/* This is mostly a test of the field order */
class BpipP8_1 extends Bundle {
  val field_1_1 = UInt(11.W)
  val field_1_2 = UInt(12.W)
}

class BpipP8_2 extends BpipP8_1 {
  val field_2_1 = UInt(11.W)
  val field_2_2 = UInt(12.W)
}

class BpipP8_3 extends BpipP8_2 {
  val field_3_1 = UInt(11.W)
  val field_3_2 = UInt(12.W)
}

/* plugin should not affect the seq detection */
class ForFieldOrderingTest extends Module {
  val out1 = IO(Output(new BpipP8_3))
  out1 := DontCare
  assertElementsMatchExpected(out1)(
    "field_3_2" -> _.field_3_2,
    "field_3_1" -> _.field_3_1,
    "field_2_2" -> _.field_2_2,
    "field_2_1" -> _.field_2_1,
    "field_1_2" -> _.field_1_2,
    "field_1_1" -> _.field_1_1
  )
}

/* plugin should allow parameter var fields */
class HasValParamsToBundle extends Module {
  // The following block does not work, suggesting that ParamIsField is not a case we need to solve
  class BpipParamIsField0(val paramField0: UInt) extends Bundle
  class BpipParamIsField1(val paramField1: UInt) extends BpipParamIsField0(UInt(66.W))

  val out3 = IO(Output(new BpipParamIsField1(UInt(10.W))))
  val out4 = IO(Output(new BpipParamIsField1(UInt(10.W))))
  out3 := DontCare
  assertElementsMatchExpected(out3)("paramField0" -> _.paramField0, "paramField1" -> _.paramField1)
  assertElementsMatchExpected(out4)("paramField0" -> _.paramField0, "paramField1" -> _.paramField1)
}

class HasGenParamsPassedToSuperclasses extends Module {

  class OtherBundle extends Bundle {
    val otherField = UInt(55.W)
  }

  class BpipWithGen[T <: Data, TT <: Data](gen: T, gen2: => TT) extends Bundle {
    val superFoo = gen
    val superQux = gen2
  }

  class BpipUsesWithGen[T <: Data](gen: T, gen2: => T) extends BpipWithGen(gen, gen2) {
    val bar = Bool()
    val qux = gen2
    val bad = 444
    val baz = Decoupled(UInt(16.W))
  }

  val out1 = IO(Output(new BpipUsesWithGen(UInt(4.W), new OtherBundle)))

  out1 := DontCare

  assertElementsMatchExpected(out1)(
    "baz" -> _.baz,
    "qux" -> _.qux,
    "bar" -> _.bar,
    "superQux" -> _.superQux,
    "superFoo" -> _.superFoo
  )
}

/* Testing whether gen fields superFoo and superQux can be found when they are
 * declared in a superclass
 *
 */
class UsesGenFiedldsInSuperClass extends Module {
  class BpipWithGen[T <: Data](gen: T) extends Bundle {
    val superFoo = gen
    val superQux = gen
  }

  class BpipUsesWithGen[T <: Data](gen: T) extends BpipWithGen(gen) {}

  val out = IO(Output(new BpipUsesWithGen(UInt(4.W))))

  out := DontCare

  assertElementsMatchExpected(out)()
}

/* Testing whether gen fields superFoo and superQux can be found when they are
 * declared in a superclass
 *
 */
class BpipBadBundleWithHardware extends Bundle {
  val bpipWithHardwareGood = UInt(8.W)
  val bpipWithHardwareBad = 244.U(16.W)
}

class HasHardwareFieldsInBundle extends Module {
  val out = IO(Output(new BpipBadBundleWithHardware))
  assertElementsMatchExpected(out)()
}

/* This is legal because of =>
 */
class UsesBundleWithGeneratorField extends Module {
  class BpipWithGen[T <: Data](gen: => T) extends Bundle {
    val superFoo = gen
    val superQux = gen
  }

  class BpipUsesWithGen[T <: Data](gen: => T) extends BpipWithGen(gen)

  val out = IO(Output(new BpipUsesWithGen(UInt(4.W))))

  out := DontCare

  assertElementsMatchExpected(out)("superQux" -> _.superQux, "superFoo" -> _.superFoo)
}

/* Testing whether gen fields superFoo and superQux can be found when they are
 * declared in a superclass
 *
 */
class BundleElementsSpec extends AnyFreeSpec with Matchers {

  /** Tests a whole bunch of different Bundle constructions
    */
  class BpipIsComplexBundle extends Module {

    trait BpipVarmint {
      val varmint = Bool()

      def vermin = Bool()

      private val puppy = Bool()
    }

    abstract class BpipAbstractBundle extends Bundle {
      def doNothing: Unit

      val fromAbstractBundle = UInt(22.W)
    }

    class BpipOneField extends Bundle {
      val fieldOne = SInt(8.W)
    }

    class BpipTwoField extends BpipOneField {
      val fieldTwo = SInt(8.W)
      val fieldThree = Vec(4, UInt(12.W))
    }

    class BpipAnimalBundle(w1: Int, w2: Int) extends Bundle {
      val dog = SInt(w1.W)
      val fox = UInt(w2.W)
    }

    class BpipDemoBundle[T <: Data](gen: T) extends BpipTwoField with BpipVarmint {
      val foo = gen
      val bar = Bool()
      val bad = 44
      val baz = Decoupled(UInt(16.W))
      val animals = new BpipAnimalBundle(4, 8)
    }

    val out = IO(Output(new BpipDemoBundle(UInt(4.W))))

    val out2 = IO(Output(new BpipAbstractBundle {
      override def doNothing: Unit = ()

      val notAbstract: Bool = Bool()
    }))

    val out4 = IO(Output(new BpipAnimalBundle(99, 100)))
    val out5 = IO(Output(new BpipTwoField))

    out := DontCare
    out5 := DontCare

    assertElementsMatchExpected(out)(
      "animals" -> _.animals,
      "baz" -> _.baz,
      "bar" -> _.bar,
      "varmint" -> _.varmint,
      "fieldThree" -> _.fieldThree,
      "fieldTwo" -> _.fieldTwo,
      "fieldOne" -> _.fieldOne,
      "foo" -> _.foo
    )
    assertElementsMatchExpected(out5)("fieldThree" -> _.fieldThree, "fieldTwo" -> _.fieldTwo, "fieldOne" -> _.fieldOne)
    assertElementsMatchExpected(out2)("notAbstract" -> _.notAbstract, "fromAbstractBundle" -> _.fromAbstractBundle)
    assertElementsMatchExpected(out4)("fox" -> _.fox, "dog" -> _.dog)
  }

  "Complex Bundle with inheritance, traits and params. DebugProblem1" in {
    ChiselStage.emitCHIRRTL(new BpipIsComplexBundle)
  }

  "Decoupled Bundle with inheritance" in {
    ChiselStage.emitCHIRRTL(new HasDecoupledBundleByInheritance)
  }

  "Simple bundle inheritance. DebugProblem3" in {
    ChiselStage.emitCHIRRTL(new DebugProblem3)
  }

  "Bundles containing Seq[Data] should be compile erorr. HasBadSeqBundle" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new HasBadSeqBundle)
    }
  }

  "IgnoreSeqInBundle allows Seq[Data] in bundle" in {
    ChiselStage.emitCHIRRTL(new UsesIgnoreSeqInBundle)
  }

  "Simple field ordering test." in {
    ChiselStage.emitCHIRRTL(new ForFieldOrderingTest)
  }

  "Val params to Bundle should be an Exception." in {
    ChiselStage.emitCHIRRTL(new HasValParamsToBundle)
  }

  "Should handle gen params passed to superclasses" in {
    ChiselStage.emitCHIRRTL(new HasGenParamsPassedToSuperclasses)
  }

  "Aliased fields should be detected and throw an exception, because gen: Data, with no =>" in {
    intercept[AliasedAggregateFieldException] {
      ChiselStage.emitCHIRRTL(new UsesGenFiedldsInSuperClass)
    }
  }

  "Error when bundle fields are hardware, such as literal values. HasHardwareFieldsInBundle" in {
    val e = intercept[ExpectedChiselTypeException] {
      ChiselStage.emitCHIRRTL(new HasHardwareFieldsInBundle)
    }
    e.getMessage should include(
      "Bundle: BpipBadBundleWithHardware contains hardware fields: bpipWithHardwareBad: UInt<16>(244)"
    )
  }

  "Aliased fields not created when using gen: => Data" in {
    ChiselStage.emitCHIRRTL(new UsesBundleWithGeneratorField)
  }

  class OptionBundle(val hasIn: Boolean) extends Bundle {
    val in = if (hasIn) {
      Some(Input(Bool()))
    } else {
      None
    }
    val out = Output(Bool())
  }

  class UsesBundleWithOptionFields extends Module {
    val outTrue = IO(Output(new OptionBundle(hasIn = true)))
    val outFalse = IO(Output(new OptionBundle(hasIn = false)))
    //NOTE: The _.in.get _.in is an optional field
    assertElementsMatchExpected(outTrue)("out" -> _.out, "in" -> _.in.get)
    assertElementsMatchExpected(outFalse)("out" -> _.out)
  }

  "plugin should work with Bundles with Option fields" in {
    ChiselStage.emitCHIRRTL(new UsesBundleWithOptionFields)
  }

  "plugin should work with enums in bundles" in {
    object Enum0 extends ChiselEnum {
      val s0, s1, s2 = Value
    }

    ChiselStage.emitCHIRRTL(new Module {
      val out = IO(Output(new Bundle {
        val a = UInt(8.W)
        val b = Bool()
        val c = Enum0.Type
      }))
      assertElementsMatchExpected(out)("b" -> _.b, "a" -> _.a)
    })
  }

  "plugin will NOT see fields that are Data but declared in some way as Any" in {
    //This is not incompatible with chisel not using the plugin, but this code is considered bad practice

    ChiselStage.emitCHIRRTL(new Module {
      val out = IO(Output(new Bundle {
        val a = UInt(8.W)
        val b: Any = Bool()
      }))

      intercept[TestFailedException] {
        assert(out.elements.keys.exists(_ == "b"))
      }
    })
  }

  "plugin tests should fail properly in the following cases" - {

    class BundleAB extends Bundle {
      val a = Output(UInt(8.W))
      val b = Output(Bool())
    }

    def checkAssertion(checks: (BundleAB => (String, Data))*)(expectedMessage: String): Unit = {
      intercept[AssertionError] {
        ChiselStage.emitCHIRRTL(new Module {
          val out = IO(new BundleAB)
          assertElementsMatchExpected(out)(checks: _*)
        })
      }.getMessage should include(expectedMessage)
    }

    "one of the expected data values is wrong" in {
      checkAssertion("b" -> _.b, "a" -> _.b)("field 'a' data field BundleElementsSpec_Anon.out.a")
    }

    "one of the expected field names in wrong" in {
      checkAssertion("b" -> _.b, "z" -> _.a)("field: 'a' did not match expected 'z'")
    }

    "fields that are expected are not returned by the elements method" in {
      checkAssertion("b" -> _.b, "a" -> _.a, "c" -> _.a)("#elements is missing the 'c' field")
    }

    "fields returned by the element are not specified in the expected fields" in {
      checkAssertion("b" -> _.b)("expected fields did not include 'a' field found in #elements")
    }

    "multiple errors between elements method and expected fields are shown in the assertion error message" in {
      checkAssertion()(
        "expected fields did not include 'b' field found in #elements," +
          " expected fields did not include 'a' field found in #elements"
      )
    }
  }

  "plugin should error correctly when bundles contain only a Option field" in {
    ChiselStage.emitCHIRRTL(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = new Bundle {
          val y = if (false) Some(Input(UInt(8.W))) else None
        }
      })
      assertElementsMatchExpected(io)("x" -> _.x, "foo" -> _.foo)
      assertElementsMatchExpected(io.x)()
    })
  }

  "plugin should handle fields using the boolean to option construct" in {
    case class ALUConfig(
      xLen: Int,
      mul:  Boolean,
      b:    Boolean)

    class OptionalBundle extends Bundle {
      val optionBundleA = Input(UInt(3.W))
      val optionBundleB = Input(UInt(7.W))
    }

    class ALU(c: ALUConfig) extends Module {

      class BpipOptionBundle extends Bundle with IgnoreSeqInBundle {
        val bpipUIntVal = Input(UInt(8.W))
        lazy val bpipUIntLazyVal = Input(UInt(8.W))
        var bpipUIntVar = Input(UInt(8.W))

        def bpipUIntDef = Input(UInt(8.W))

        val bpipOptionUInt = Some(Input(UInt(16.W)))
        val bpipOptionOfBundle = if (c.b) Some(new OptionalBundle) else None
        val bpipSeqData = Seq(UInt(8.W), UInt(8.W))
      }

      val io = IO(new BpipOptionBundle)
      assertElementsMatchExpected(io)(
        "bpipUIntLazyVal" -> _.bpipUIntLazyVal,
        "bpipOptionUInt" -> _.bpipOptionUInt.get,
        "bpipUIntVar" -> _.bpipUIntVar,
        "bpipUIntVal" -> _.bpipUIntVal
      )
    }

    ChiselStage.emitCHIRRTL(new ALU(ALUConfig(10, mul = true, b = false)))
  }

  "TraceSpec test, different version found in TraceSpec.scala" in {
    class Bundle0 extends Bundle {
      val a = UInt(8.W)
      val b = Bool()
      val c = Enum0.Type
    }

    class Bundle1 extends Bundle {
      val a = new Bundle0
      val b = Vec(4, Vec(4, Bool()))
    }

    class Module0 extends Module {
      val i = IO(Input(new Bundle1))
      val o = IO(Output(new Bundle1))
      val r = Reg(new Bundle1)
      o := r
      r := i

      assertElementsMatchExpected(i)("b" -> _.b, "a" -> _.a)
      assertElementsMatchExpected(o)("b" -> _.b, "a" -> _.a)
      assertElementsMatchExpected(r)("b" -> _.b, "a" -> _.a)
    }

    class Module1 extends Module {
      val i = IO(Input(new Bundle1))
      val m0 = Module(new Module0)
      m0.i := i
      m0.o := DontCare
      assertElementsMatchExpected(i)("b" -> _.b, "a" -> _.a)
    }

    object Enum0 extends ChiselEnum {
      val s0, s1, s2 = Value
    }

    ChiselStage.emitCHIRRTL(new Module1)
  }
}
/* Checks that the elements method of a bundle matches the testers idea of what the bundle field names and their
 * associated data match and that they have the same number of fields in the same order
 */
object assertElementsMatchExpected {
  def apply[T <: Bundle](bun: T)(checks: (T => (String, Data))*): Unit = {
    val expected = checks.map { fn => fn(bun) }
    val elements = bun.elements
    val missingMsg = "missing field in #elements"
    val extraMsg = "extra field in #elements"
    val paired = elements.toSeq.zipAll(expected, missingMsg -> UInt(1.W), extraMsg -> UInt(1.W))
    val errorsStrings = paired.flatMap {
      case (element, expected) =>
        val (elementName, elementData) = element
        val (expectedName, expectedData) = expected
        if (elementName == missingMsg) {
          Some(s"#elements is missing the '$expectedName' field")
        } else if (expectedName == extraMsg) {
          Some(s"expected fields did not include '$elementName' field found in #elements")
        } else if (elementName != expectedName) {
          Some(s"field: '$elementName' did not match expected '$expectedName'")
        } else if (elementData != expectedData) {
          Some(
            s"field '$elementName' data field ${elementData}(${elementData.hashCode}) did not match expected $expectedData(${expectedData.hashCode})"
          )
        } else {
          None
        }
    }
    assert(errorsStrings.isEmpty, s"Bundle: ${bun.getClass.getName}: " + errorsStrings.mkString(", "))
  }
}
