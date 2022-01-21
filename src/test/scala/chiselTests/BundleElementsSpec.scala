// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.{ChiselEnum, FixedPoint}
import chisel3.stage.ChiselStage
import chisel3.util.Decoupled
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

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
  assert(
    FieldsMatch(
      out1,
      "bpipDecoupledDecoupled",
      "bpipDecoupledVec",
      "bpipDecoupledSInt",
      "bpipOneFieldTwo",
      "bpipOneFieldOne",
      "bpipTraitGood",
      "bpipSuperTraitGood"
    )
  )
}

/* plugin should not affect the seq detection */
class DebugProblem3 extends Module {
  val out1 = IO(Output(new BpipTwoField))
  assert(
    FieldsMatch(
      out1,
      "baz",
      "bpipTwoFieldTwo",
      "bpipTwoFieldOne",
      "bpipOneFieldTwo",
      "bpipOneFieldOne",
      "bpipTraitGood",
      "bpipSuperTraitGood"
    )
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
  assert(FieldsMatch(out1, "field_3_2", "field_3_1", "field_2_2", "field_2_1", "field_1_2", "field_1_1"))
}

/* plugin should allow parameter var fields */
class HasValParamsToBundle extends Module {
  // The following block does not work, suggesting that ParamIsField is not a case we need to solve
  class BpipParamIsField0(val paramField0: UInt) extends Bundle
  class BpipParamIsField1(val paramField1: UInt) extends BpipParamIsField0(UInt(66.W))

  val out3 = IO(Output(new BpipParamIsField1(UInt(10.W))))
  val out4 = IO(Output(new BpipParamIsField1(UInt(10.W))))
  out3 := DontCare
  assert(FieldsMatch(out3, "paramField0", "paramField1"))
  assert(FieldsMatch(out4, "paramField0", "paramField1"))
}

class HasGenParamsPassedToSuperclasses extends Module {

  class OtherBundle extends Bundle {
    val otherField = UInt(55.W)
  }

  class BpipWithGen[T <: Data, TT <: Data](gen: T, gen2: => TT) extends Bundle {
    val superFoo = gen
    val superQux = gen2
  }

//  class BpipDemoBundle[T <: Data](gen: T, gen2: => T) extends BpipTwoField with BpipVarmint {
  class BpipUsesWithGen[T <: Data](gen: T, gen2: => T) extends BpipWithGen(gen, gen2) {
    //    val foo = gen
    val bar = Bool()
    val qux = gen2
    val bad = 444
    val baz = Decoupled(UInt(16.W))
  }

  val out1 = IO(Output(new BpipUsesWithGen(UInt(4.W), new OtherBundle)))
  val out2 = IO(Output(new BpipUsesWithGen(UInt(4.W), FixedPoint(10.W, 4.BP))))

  out1 := DontCare

  assert(FieldsMatch(out1, "baz", "qux", "bar", "superQux", "superFoo"))
  assert(FieldsMatch(out2, "baz", "qux", "bar", "superQux", "superFoo"))
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

  assert(FieldsMatch(out))
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
  assert(FieldsMatch(out))
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

  assert(FieldsMatch(out, "superQux", "superFoo"))
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

    class BpipDemoBundle[T <: Data](gen: T, gen2: => T) extends BpipTwoField with BpipVarmint {
      val foo = gen
      val bar = Bool()
      val qux = gen2
      val bad = 44
      val baz = Decoupled(UInt(16.W))
      val animals = new BpipAnimalBundle(4, 8)
    }

    val out = IO(Output(new BpipDemoBundle(UInt(4.W), FixedPoint(10.W, 4.BP))))

    val out2 = IO(Output(new BpipAbstractBundle {
      override def doNothing: Unit = println("ugh")
      val notAbstract:        Bool = Bool()
    }))

    val out4 = IO(Output(new BpipAnimalBundle(99, 100)))
    val out5 = IO(Output(new BpipTwoField))

    out := DontCare
    out5 := DontCare

    assert(FieldsMatch(out, "animals", "baz", "qux", "bar", "varmint", "fieldThree", "fieldTwo", "fieldOne", "foo"))
    assert(FieldsMatch(out5, "fieldThree", "fieldTwo", "fieldOne"))
    assert(FieldsMatch(out2, "notAbstract", "fromAbstractBundle"))
    assert(FieldsMatch(out4, "fox", "dog"))
  }

  "Complex Bundle with inheritance, traits and params. DebugProblem1" in {
    ChiselStage.emitFirrtl(new BpipIsComplexBundle)
  }

  "Decoupled Bundle with inheritance" in {
    ChiselStage.emitFirrtl(new HasDecoupledBundleByInheritance)
  }

  "Simple bundle inheritance. DebugProblem3" in {
    ChiselStage.emitFirrtl(new DebugProblem3)
  }

  "Bundles containing Seq[Data] should be compile erorr. HasBadSeqBundle" in {
    intercept[ChiselException] {
      ChiselStage.emitFirrtl(new HasBadSeqBundle)
    }
  }

  "IgnoreSeqInBundle allows Seq[Data] in bundle" in {
    ChiselStage.emitFirrtl(new UsesIgnoreSeqInBundle)
  }

  "Simple field ordering test." in {
    ChiselStage.emitFirrtl(new ForFieldOrderingTest)
  }

  "Val params to Bundle should be an Exception." in {
    ChiselStage.emitFirrtl(new HasValParamsToBundle)
  }

  "Should handle gen params passed to superclasses" in {
    ChiselStage.emitFirrtl(new HasGenParamsPassedToSuperclasses)
  }

  "Aliased fields should be detected and throw an exception, because gen: Data, with no =>" in {
    intercept[AliasedAggregateFieldException] {
      ChiselStage.emitFirrtl(new UsesGenFiedldsInSuperClass)
    }
  }

  "Error when bundle fields are hardware, such as literal values. HasHardwareFieldsInBundle" in {
    val e = intercept[ExpectedChiselTypeException] {
      ChiselStage.emitFirrtl(new HasHardwareFieldsInBundle)
    }
    e.getMessage should include(
      "Bundle: BpipBadBundleWithHardware contains hardware fields: bpipWithHardwareBad: UInt<16>(244)"
    )
  }

  "Aliased fields not created when using gen: => Data" in {
    ChiselStage.emitFirrtl(new UsesBundleWithGeneratorField)
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
    assert(FieldsMatch(outTrue, "out", "in"))
    assert(FieldsMatch(outFalse, "out"))
  }

  "plugin should work with Bundles with Option fields" in {
    ChiselStage.emitFirrtl(new UsesBundleWithOptionFields)
  }

  "plugin should work with enums in bundles" in {

    object Enum0 extends ChiselEnum {
      val s0, s1, s2 = Value
    }

    ChiselStage.emitFirrtl(new Module {
      val out = IO(Output(new Bundle {
        val a = UInt(8.W)
        val b = Bool()
        val c = Enum0.Type
      }))
      assert(FieldsMatch(out, "b", "a"))
    })
  }

  "plugin should error correctly when bundles contain only a Option field" in {
    ChiselStage.emitFirrtl(new Module {
      val io = IO(new Bundle {
        val foo = Input(UInt(8.W))
        val x = new Bundle {
          val y = if (false) Some(Input(UInt(8.W))) else None
        }
      })
      assert(FieldsMatch(io, "x", "foo"))
      assert(FieldsMatch(io.x))
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
      assert(FieldsMatch(io, "bpipUIntLazyVal", "bpipOptionUInt", "bpipUIntVar", "bpipUIntVal"))
    }

    ChiselStage.emitFirrtl(new ALU(ALUConfig(10, mul = true, b = false)))
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

      assert(FieldsMatch(i, "b", "a"))
      assert(FieldsMatch(o, "b", "a"))
      assert(FieldsMatch(r, "b", "a"))
    }

    class Module1 extends Module {
      val i = IO(Input(new Bundle1))
      val m0 = Module(new Module0)
      m0.i := i
      m0.o := DontCare
      assert(FieldsMatch(i, "b", "a"))

    }

    object Enum0 extends ChiselEnum {
      val s0, s1, s2 = Value
    }

    ChiselStage.emitFirrtl(new Module1)
  }
}

object FieldsMatch {
  def apply(bundle: Bundle, fieldNames: String*): Boolean = {
    fieldNames.toSeq
      .zipAll(bundle.elements.map(_._1).toList, "Missing Bundle Field", "Extra BundleField")
      .forall {
        case (fromBundle, fromList) =>
          if (fromBundle != fromList) {
            println(s"""Bundle: ${bundle}, field mismatch "$fromBundle" to "$fromList" """)
          }
          fromBundle == fromList
      }
  }
}
