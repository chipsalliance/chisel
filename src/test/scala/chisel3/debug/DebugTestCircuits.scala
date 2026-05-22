// SPDX-License-Identifier: Apache-2.0

package chisel3.debug

import chisel3._
import chisel3.experimental.{fromStringToStringParam, Analog}
import chisel3.util.MixedVec
import circt.stage.ChiselStage
import org.scalatest.matchers.should.Matchers

object DebugTestEnum extends ChiselEnum { val EA, EB, EC = Value }
object DebugTestEnum2 extends ChiselEnum { val ED, EE, EF = Value }

object DebugTestUtils extends Matchers {

  private def q(s: String): String = java.util.regex.Pattern.quote(s)

  def countOccurrences(chirrtl: String, pattern: String): Int =
    pattern.r.findAllMatchIn(chirrtl).length

  def getMissingOccurrences(chirrtl: String, pattern: String, expectedLines: Seq[String]): String = {
    val lines = chirrtl.split("\n").toSeq
    val anyR = (".*" + pattern + ".*").r
    val idxs = lines.indices.filter(i => anyR.findFirstIn(lines(i)).isDefined)
    val withNext =
      idxs.flatMap(i => if (i < lines.length - 1) List(lines(i), lines(i + 1)) else List(lines(i))).distinct
    val expRe = expectedLines.map(_.r)
    withNext.filterNot(line => expRe.exists(_.findFirstIn(line).isDefined)).mkString("\n")
  }

  private def attrPart(name: String, value: String): String = s"""[^)]*$name\\s*=\\s*"${q(value)}""""
  private def enumPart(et:   Option[String]):        String = et.fold("")(attrPart("enumTypeName", _))

  private def intrinsic(kind: String, attrs: String*): String =
    s"""intrinsic\\(circt_debug_$kind<${attrs.mkString}"""

  def varPattern(typeName: String, name: String, enumTypeName: Option[String] = None): String =
    intrinsic("var", attrPart("typeName", typeName), attrPart("name", name), enumPart(enumTypeName))

  def subfieldPattern(typeName: String, name: String, parent: String, enumTypeName: Option[String] = None): String =
    intrinsic(
      "subfield",
      attrPart("typeName", typeName),
      attrPart("name", name),
      attrPart("parent", parent),
      enumPart(enumTypeName)
    )

  def enumDefPattern(typeName:    String): String = intrinsic("enumdef", attrPart("typeName", typeName))
  def moduleInfoPattern(typeName: String): String = intrinsic("moduleinfo", attrPart("typeName", typeName))

  // Collapse whitespace so regex helpers ignore line wrapping.
  def emit(gen: => RawModule): String =
    ChiselStage.emitCHIRRTL(gen, args = Array("--with-experimental-debug-intrinsics")).replaceAll("\\s+", " ")

  private val moduleInfoBodyRe = """intrinsic\(circt_debug_moduleinfo<(.*?)>\)""".r
  private val paramsBodyRe = """params\s*=\s*"((?:[^"\\]|\\.)*)"""".r

  def requireParams(chirrtl: String, typeName: String): String = {
    val tnPat = s"""typeName\\s*=\\s*"${q(typeName)}"""".r
    moduleInfoBodyRe
      .findAllMatchIn(chirrtl)
      .map(_.group(1))
      .find(body => tnPat.findFirstIn(body).isDefined)
      .flatMap(body => paramsBodyRe.findFirstMatchIn(body).map(_.group(1)))
      .getOrElse(fail(s"no circt_debug_moduleinfo for $typeName"))
  }

  def assertParam(params: String, name: String, typeName: String, value: Any): Unit = {
    val bq = "\\\""
    val valueJson = value match {
      case b: Boolean => b.toString
      case other => bq + other.toString + bq
    }
    params should include(s"${bq}name$bq:$bq$name$bq")
    params should include(s"${bq}typeName$bq:$bq$typeName$bq")
    params should include(s"${bq}value$bq:$valueJson")
  }

  def checkIntrinsics(expected: Seq[(String, Int)], chirrtl: String): Unit = {
    val names = expected.map(_._1)
    expected.foreach { case (pattern, count) =>
      withClue(s"Pattern '$pattern':\n${getMissingOccurrences(chirrtl, pattern, names)}\n") {
        countOccurrences(chirrtl, pattern) should be(count)
      }
    }
  }
}

object DebugTestCircuits {

  sealed abstract class BindingChoice(label: String) {
    def apply[T <: Data](data: T): T
    override def toString = label
  }
  case object PortBinding extends BindingChoice("IO") { def apply[T <: Data](d: T): T = IO(d) }
  case object WireBinding extends BindingChoice("Wire") { def apply[T <: Data](d: T): T = Wire(d) }
  case object RegBinding extends BindingChoice("Reg") { def apply[T <: Data](d: T): T = Reg(d) }

  abstract class DebugTestModule(bindingChoice: BindingChoice) extends RawModule {
    def body: Unit
    bindingChoice match {
      case RegBinding =>
        val clock = IO(Input(Clock()))
        val reset = IO(Input(Reset()))
        withClockAndReset(clock, reset) { body }
      case _ => body
    }
  }

  object ModuleCircuits {

    class TopCircuitBlackBox extends RawModule {
      class MyBlackBox extends ExtModule(Map("PARAM1" -> "TRUE", "PARAM2" -> "DEFAULT")) {
        val io = IO(new Bundle {})
      }
      val myBlackBox1:  MyBlackBox = Module(new MyBlackBox)
      val myBlackBox2:  MyBlackBox = Module(new MyBlackBox)
      val myBlackBoxes: Seq[MyBlackBox] = Seq.fill(2)(Module(new MyBlackBox))
    }
  }

  object DataTypesCircuits {

    class TopCircuitClockReset extends RawModule {
      val clock:      Clock = IO(Input(Clock()))
      val syncReset:  Bool = IO(Input(Bool()))
      val reset:      Reset = IO(Input(Reset()))
      val asyncReset: AsyncReset = IO(Input(AsyncReset()))
    }

    class TopCircuitImplicitClockReset extends Module

    class TopCircuitGroundTypes(b: BindingChoice) extends DebugTestModule(b) {
      override def body: Unit = {
        val uint: UInt = b(UInt(8.W))
        val sint: SInt = b(SInt(8.W))
        val bool: Bool = b(Bool())
        if (b != RegBinding) {
          val analog: Analog = b(Analog(1.W))
        }
        val bits: UInt = b(Bits(8.W))
      }
    }

    class MyEmptyBundle extends Bundle

    class MyBundle extends Bundle {
      val a: UInt = UInt(8.W)
      val b: SInt = SInt(8.W)
      val c: Bool = Bool()
    }

    class TopCircuitBundles(b: BindingChoice) extends DebugTestModule(b) {
      override def body: Unit = {
        val a:   Bundle = b(new Bundle {})
        val bnd: MyEmptyBundle = b(new MyEmptyBundle)
        val c:   MyBundle = b(new MyBundle)
      }
    }

    class MyNestedBundle extends Bundle {
      val a: Bool = Bool()
      val b: MyBundle = new MyBundle
      val c: MyBundle = Flipped(new MyBundle)
    }

    class TopCircuitBundlesNested(b: BindingChoice) extends DebugTestModule(b) {
      override def body: Unit = {
        val a: MyNestedBundle = b(new MyNestedBundle)
      }
    }

    class TopCircuitVecs(b: BindingChoice) extends DebugTestModule(b) {
      override def body: Unit = {
        val a:  Vec[SInt] = b(Vec(5, SInt(23.W)))
        val bv: Vec[Vec[SInt]] = b(Vec(5, Vec(3, SInt(23.W))))
        val c = b(Vec(5, new Bundle { val x: UInt = UInt(8.W) }))
        val d = b(MixedVec(UInt(3.W), SInt(10.W)))
      }
    }

    class TopCircuitBundleWithVec(b: BindingChoice) extends DebugTestModule(b) {
      override def body: Unit = {
        val a = b(new Bundle { val vec = Vec(5, UInt(8.W)) })
      }
    }

    class TopCircuitWhenElse extends RawModule {
      val inSeq = IO(Input(Vec(8, UInt(8.W))))
      val out = IO(Output(UInt(8.W)))
      val sel = IO(Input(UInt(math.sqrt(8).ceil.toInt.W)))
      val tmp = sel + 1.U
      when(sel % 2.U === 0.U) {
        val outTmp = inSeq(sel)
        val evenSel = outTmp + 1.U
        out := evenSel
      }.elsewhen(sel === 1.U) {
        val outTmp = inSeq(sel)
        val selIsOne = outTmp + 1.U
        out := selIsOne
      }.otherwise {
        val outTmp = inSeq(sel)
        val oddSel = outTmp + 1.U
        out := oddSel
      }
    }

    class MyDeeplyNestedBundle extends Bundle {
      val a = new Bundle {
        val b = new Bundle {
          val c: UInt = UInt(8.W)
        }
      }
    }

    class TopCircuitDeeplyNested extends RawModule {
      val io = IO(Input(new MyDeeplyNestedBundle))
    }

    class TopCircuitEnumSimple extends RawModule {
      val e = IO(Input(DebugTestEnum()))
      val e2 = IO(Input(DebugTestEnum2()))
      val bnd = IO(Input(new Bundle {
        val en = DebugTestEnum()
        val en2 = DebugTestEnum2()
        val x = UInt(8.W)
      }))
      val v = IO(Input(Vec(3, DebugTestEnum())))
    }

  }

  object MemCircuits {
    class TopCircuitMem[T <: Data](gen: T) extends Module { val mem = Mem(4, gen) }
    class TopCircuitSyncMem[T <: Data](gen: T) extends Module { val mem = SyncReadMem(4, gen) }
  }
}
