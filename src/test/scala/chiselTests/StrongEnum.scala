// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.ChiselEnum
import chisel3.internal.firrtl.UnknownWidth
import chisel3.internal.naming.chiselName
import chisel3.stage.{ChiselGeneratorAnnotation, ChiselStage}
import chisel3.util._
import chisel3.testers.BasicTester
import org.scalatest.Assertion
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers

object EnumExample extends ChiselEnum {
  val e0, e1, e2 = Value

  val e100 = Value(100.U)
  val e101 = Value(101.U)

  val litValues = List(0.U, 1.U, 2.U, 100.U, 101.U)
}

object OtherEnum extends ChiselEnum {
  val otherEnum = Value
}

object NonLiteralEnumType extends ChiselEnum {
  val nonLit = Value(UInt())
}

object NonIncreasingEnum extends ChiselEnum {
  val x = Value(2.U)
  val y = Value(2.U)
}

class SimpleConnector(inType: Data, outType: Data) extends Module {
  val io = IO(new Bundle {
    val in = Input(inType)
    val out = Output(outType)
  })

  io.out := io.in
}

class CastToUInt extends Module {
  val io = IO(new Bundle {
    val in = Input(EnumExample())
    val out = Output(UInt())
  })

  io.out := io.in.asUInt()
}

class CastFromLit(in: UInt) extends Module {
  val io = IO(new Bundle {
    val out = Output(EnumExample())
    val valid = Output(Bool())
  })

  io.out := EnumExample(in)
  io.valid := io.out.isValid
}

class CastFromNonLit extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(EnumExample.getWidth.W))
    val out = Output(EnumExample())
    val valid = Output(Bool())
  })

  io.out := EnumExample(io.in)
  io.valid := io.out.isValid
}

class CastFromNonLitWidth(w: Option[Int] = None) extends Module {
  val width = if (w.isDefined) w.get.W else UnknownWidth()

  override val io = IO(new Bundle {
    val in = Input(UInt(width))
    val out = Output(EnumExample())
  })

  io.out := EnumExample(io.in)
}

class EnumOps(val xType: ChiselEnum, val yType: ChiselEnum) extends Module {
  val io = IO(new Bundle {
    val x = Input(xType())
    val y = Input(yType())

    val lt = Output(Bool())
    val le = Output(Bool())
    val gt = Output(Bool())
    val ge = Output(Bool())
    val eq = Output(Bool())
    val ne = Output(Bool())
  })

  io.lt := io.x < io.y
  io.le := io.x <= io.y
  io.gt := io.x > io.y
  io.ge := io.x >= io.y
  io.eq := io.x === io.y
  io.ne := io.x =/= io.y
}

object StrongEnumFSM {
  object State extends ChiselEnum {
    val sNone, sOne1, sTwo1s = Value

    val correct_annotation_map = Map[String, BigInt]("sNone" -> 0, "sOne1" -> 1, "sTwo1s" -> 2)
  }
}

class StrongEnumFSM extends Module {
  import StrongEnumFSM.State
  import StrongEnumFSM.State._

  // This FSM detects two 1's one after the other
  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
    val state = Output(State())
  })

  val state = RegInit(sNone)

  io.out := (state === sTwo1s)
  io.state := state

  switch (state) {
    is (sNone) {
      when (io.in) {
        state := sOne1
      }
    }
    is (sOne1) {
      when (io.in) {
        state := sTwo1s
      } .otherwise {
        state := sNone
      }
    }
    is (sTwo1s) {
      when (!io.in) {
        state := sNone
      }
    }
  }
}

class CastToUIntTester extends BasicTester {
  for ((enum,lit) <- EnumExample.all zip EnumExample.litValues) {
    val mod = Module(new CastToUInt)
    mod.io.in := enum
    assert(mod.io.out === lit)
  }
  stop()
}

class CastFromLitTester extends BasicTester {
  for ((enum,lit) <- EnumExample.all zip EnumExample.litValues) {
    val mod = Module(new CastFromLit(lit))
    assert(mod.io.out === enum)
    assert(mod.io.valid === true.B)
  }
  stop()
}

class CastFromNonLitTester extends BasicTester {
  for ((enum,lit) <- EnumExample.all zip EnumExample.litValues) {
    val mod = Module(new CastFromNonLit)
    mod.io.in := lit
    assert(mod.io.out === enum)
    assert(mod.io.valid === true.B)
  }

  val invalid_values = (1 until (1 << EnumExample.getWidth)).
    filter(!EnumExample.litValues.map(_.litValue).contains(_)).
    map(_.U)

  for (invalid_val <- invalid_values) {
    val mod = Module(new CastFromNonLit)
    mod.io.in := invalid_val

    assert(mod.io.valid === false.B)
  }

  stop()
}

class CastToInvalidEnumTester extends BasicTester {
  val invalid_value: UInt = EnumExample.litValues.last + 1.U
  Module(new CastFromLit(invalid_value))
}

class EnumOpsTester extends BasicTester {
  for (x <- EnumExample.all;
       y <- EnumExample.all) {
    val mod = Module(new EnumOps(EnumExample, EnumExample))
    mod.io.x := x
    mod.io.y := y

    assert(mod.io.lt === (x.asUInt() < y.asUInt()))
    assert(mod.io.le === (x.asUInt() <= y.asUInt()))
    assert(mod.io.gt === (x.asUInt() > y.asUInt()))
    assert(mod.io.ge === (x.asUInt() >= y.asUInt()))
    assert(mod.io.eq === (x.asUInt() === y.asUInt()))
    assert(mod.io.ne === (x.asUInt() =/= y.asUInt()))
  }
  stop()
}

class InvalidEnumOpsTester extends BasicTester {
  val mod = Module(new EnumOps(EnumExample, OtherEnum))
  mod.io.x := EnumExample.e0
  mod.io.y := OtherEnum.otherEnum
}

class IsLitTester extends BasicTester {
  for (e <- EnumExample.all) {
    val wire = WireDefault(e)

    assert(e.isLit())
    assert(!wire.isLit())
  }
  stop()
}

class NextTester extends BasicTester {
  for ((e,n) <- EnumExample.all.zip(EnumExample.litValues.tail :+ EnumExample.litValues.head)) {
    assert(e.next.litValue == n.litValue)
    val w = WireDefault(e)
    assert(w.next === EnumExample(n))
  }
  stop()
}

class WidthTester extends BasicTester {
  assert(EnumExample.getWidth == EnumExample.litValues.last.getWidth)
  assert(EnumExample.all.forall(_.getWidth == EnumExample.litValues.last.getWidth))
  assert(EnumExample.all.forall{e =>
    val w = WireDefault(e)
    w.getWidth == EnumExample.litValues.last.getWidth
  })
  stop()
}

class StrongEnumFSMTester extends BasicTester {
  import StrongEnumFSM.State._

  val dut = Module(new StrongEnumFSM)

  // Inputs and expected results
  val inputs: Vec[Bool] = VecInit(false.B, true.B, false.B, true.B, true.B, true.B, false.B, true.B, true.B, false.B)
  val expected: Vec[Bool] = VecInit(false.B, false.B, false.B, false.B, false.B, true.B, true.B, false.B, false.B, true.B) // scalastyle:ignore line.size.limit
  val expected_state = VecInit(sNone, sNone, sOne1, sNone, sOne1, sTwo1s, sTwo1s, sNone, sOne1, sTwo1s)

  val cntr = Counter(inputs.length)
  val cycle = cntr.value

  dut.io.in := inputs(cycle)
  assert(dut.io.out === expected(cycle))
  assert(dut.io.state === expected_state(cycle))

  when(cntr.inc()) {
    stop()
  }
}

class StrongEnumSpec extends ChiselFlatSpec with Utils {
  import chisel3.internal.ChiselException

  behavior of "Strong enum tester"

  it should "fail to instantiate non-literal enums with the Value function" in {
    an [ExceptionInInitializerError] should be thrownBy extractCause[ExceptionInInitializerError] {
      ChiselStage.elaborate(new SimpleConnector(NonLiteralEnumType(), NonLiteralEnumType()))
    }
  }

  it should "fail to instantiate non-increasing enums with the Value function" in {
    an [ExceptionInInitializerError] should be thrownBy extractCause[ExceptionInInitializerError]  {
      ChiselStage.elaborate(new SimpleConnector(NonIncreasingEnum(), NonIncreasingEnum()))
    }
  }

  it should "connect enums of the same type" in {
    ChiselStage.elaborate(new SimpleConnector(EnumExample(), EnumExample()))
    ChiselStage.elaborate(new SimpleConnector(EnumExample(), EnumExample.Type()))
  }

  it should "fail to connect a strong enum to a UInt" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new SimpleConnector(EnumExample(), UInt()))
    }
  }

  it should "fail to connect enums of different types" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new SimpleConnector(EnumExample(), OtherEnum()))
    }

    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new SimpleConnector(EnumExample.Type(), OtherEnum.Type()))
    }
  }

  it should "cast enums to UInts correctly" in {
    assertTesterPasses(new CastToUIntTester)
  }

  it should "cast literal UInts to enums correctly" in {
    assertTesterPasses(new CastFromLitTester)
  }

  it should "cast non-literal UInts to enums correctly and detect illegal casts" in {
    assertTesterPasses(new CastFromNonLitTester)
  }

  it should "prevent illegal literal casts to enums" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new CastToInvalidEnumTester)
    }
  }

  it should "only allow non-literal casts to enums if the width is smaller than or equal to the enum width" in {
    for (w <- 0 to EnumExample.getWidth)
      ChiselStage.elaborate(new CastFromNonLitWidth(Some(w)))

    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new CastFromNonLitWidth)
    }

    for (w <- (EnumExample.getWidth + 1) to (EnumExample.getWidth + 100)) {
      a [ChiselException] should be thrownBy extractCause[ChiselException] {
        ChiselStage.elaborate(new CastFromNonLitWidth(Some(w)))
      }
    }
  }

  it should "execute enum comparison operations correctly" in {
    assertTesterPasses(new EnumOpsTester)
  }

  it should "fail to compare enums of different types" in {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new InvalidEnumOpsTester)
    }
  }

  it should "correctly check whether or not enums are literal" in {
    assertTesterPasses(new IsLitTester)
  }

  it should "return the correct next values for enums" in {
    assertTesterPasses(new NextTester)
  }

  it should "return the correct widths for enums" in {
    assertTesterPasses(new WidthTester)
  }

  it should "maintain Scala-level type-safety" in {
    def foo(e: EnumExample.Type): Unit = {}

    "foo(EnumExample.e1); foo(EnumExample.e1.next)" should compile
    "foo(OtherEnum.otherEnum)" shouldNot compile
  }

  it should "prevent enums from being declared without names" in {
    "object UnnamedEnum extends ChiselEnum { Value }" shouldNot compile
  }

  "StrongEnum FSM" should "work" in {
    assertTesterPasses(new StrongEnumFSMTester)
  }
}

class StrongEnumAnnotator extends Module {
  import EnumExample._

  object LocalEnum extends ChiselEnum {
    val le0, le1 = Value
    val le2 = Value
    val le100 = Value(100.U)
  }

  val io = IO(new Bundle{
    val in = Input(EnumExample())
    val out = Output(EnumExample())
    val other = Output(OtherEnum())
    val local = Output(LocalEnum())
  })

  class Bund extends Bundle {
    val field = EnumExample()
    val other = OtherEnum()
    val local = LocalEnum()
    val vec = Vec(5, EnumExample())
    val inner_bundle1 = new Bundle {
      val x = UInt(4.W)
      val y = Vec(3, UInt(4.W))
      val e = EnumExample()
      val v = Vec(3, EnumExample())
    }
    val inner_bundle2 = new Bundle {}
    val inner_bundle3 = new Bundle {
      val x = Bool()
    }
    val inner_bundle4 = new Bundle {
      val inner_inner_bundle = new Bundle {}
    }
  }

  val simple = Wire(EnumExample())
  val vec = VecInit(e0, e1, e2)
  val vec_of_vecs = VecInit(VecInit(e0, e1), VecInit(e100, e101))

  val bund = Wire(new Bund())
  val vec_of_bundles = Wire(Vec(5, new Bund()))

  io.out := e101
  io.other := OtherEnum.otherEnum
  io.local := LocalEnum.le0
  simple := e100
  bund := DontCare
  vec_of_bundles := DontCare

  // Make sure that dynamically indexing into a Vec of enums will not cause an elaboration error.
  // The components created here will not be annotated.
  val cycle = RegInit(0.U)
  cycle := cycle + 1.U

  val indexed1 = vec_of_vecs(cycle)(cycle)
  val indexed2 = vec_of_bundles(cycle)
}

@chiselName
class StrongEnumAnnotatorWithChiselName extends Module {
  import EnumExample._

  object LocalEnum extends ChiselEnum {
    val le0, le1 = Value
    val le2 = Value
    val le100 = Value(100.U)
  }

  val io = IO(new Bundle{
    val in = Input(EnumExample())
    val out = Output(EnumExample())
    val other = Output(OtherEnum())
    val local = Output(LocalEnum())
  })

  class Bund extends Bundle {
    val field = EnumExample()
    val other = OtherEnum()
    val local = LocalEnum()
    val vec = Vec(5, EnumExample())
    val inner_bundle1 = new Bundle {
      val x = UInt(4.W)
      val y = Vec(3, UInt(4.W))
      val e = EnumExample()
      val v = Vec(3, EnumExample())
    }
    val inner_bundle2 = new Bundle {}
    val inner_bundle3 = new Bundle {
      val x = Bool()
    }
    val inner_bundle4 = new Bundle {
      val inner_inner_bundle = new Bundle {}
    }
  }

  val simple = Wire(EnumExample())
  val vec = VecInit(e0, e1, e2)
  val vec_of_vecs = VecInit(VecInit(e0, e1), VecInit(e100, e101))

  val bund = Wire(new Bund())
  val vec_of_bundles = Wire(Vec(5, new Bund()))

  io.out := e101
  io.other := OtherEnum.otherEnum
  io.local := LocalEnum.le0
  simple := e100
  bund := DontCare
  vec_of_bundles := DontCare

  // Make sure that dynamically indexing into a Vec of enums will not cause an elaboration error.
  // The components created here will not be annotated.
  val cycle = RegInit(0.U)
  cycle := cycle + 1.U

  val indexed1 = vec_of_vecs(cycle)(cycle)
  val indexed2 = vec_of_bundles(cycle)
}

class StrongEnumAnnotationSpec extends AnyFreeSpec with Matchers {
  import chisel3.experimental.EnumAnnotations._
  import firrtl.annotations.{ComponentName, Annotation}

  val enumExampleName = "EnumExample"
  val otherEnumName = "OtherEnum"
  val localEnumName = "LocalEnum"

  case class CorrectDefAnno(typeName: String, definition: Map[String, BigInt])
  case class CorrectCompAnno(targetName: String, typeName: String)
  case class CorrectVecAnno(targetName: String, typeName: String, fields: Set[Seq[String]])

  val correctDefAnnos = Seq(
    CorrectDefAnno(otherEnumName, Map("otherEnum" -> 0)),
    CorrectDefAnno(enumExampleName, Map("e0" -> 0, "e1" -> 1, "e2" -> 2, "e100" -> 100, "e101" -> 101)),
    CorrectDefAnno(localEnumName, Map("le0" -> 0, "le1" -> 1, "le2" -> 2, "le100" -> 100))
  )

  val correctCompAnnos = Seq(
    CorrectCompAnno("io.other", otherEnumName),
    CorrectCompAnno("io.local", localEnumName),
    CorrectCompAnno("io.out", enumExampleName),
    CorrectCompAnno("io.in", enumExampleName),
    CorrectCompAnno("simple", enumExampleName),
    CorrectCompAnno("bund.field", enumExampleName),
    CorrectCompAnno("bund.other", otherEnumName),
    CorrectCompAnno("bund.local", localEnumName),
    CorrectCompAnno("bund.inner_bundle1.e", enumExampleName)
  )

  val correctVecAnnos = Seq(
    CorrectVecAnno("vec", enumExampleName, Set()),
    CorrectVecAnno("vec_of_vecs", enumExampleName, Set()),
    CorrectVecAnno("vec_of_bundles", enumExampleName, Set(Seq("field"), Seq("vec"), Seq("inner_bundle1", "e"), Seq("inner_bundle1", "v"))),
    CorrectVecAnno("vec_of_bundles", otherEnumName, Set(Seq("other"))),
    CorrectVecAnno("vec_of_bundles", localEnumName, Set(Seq("local"))),
    CorrectVecAnno("bund.vec", enumExampleName, Set()),
    CorrectVecAnno("bund.inner_bundle1.v", enumExampleName, Set())
  )

  // scalastyle:off regex
  def printAnnos(annos: Seq[Annotation]) {
    println("Enum definitions:")
    annos.foreach {
      case EnumDefAnnotation(enumTypeName, definition) => println(s"\t$enumTypeName: $definition")
      case _ =>
    }
    println("Enum components:")
    annos.foreach{
      case EnumComponentAnnotation(target, enumTypeName) => println(s"\t$target => $enumTypeName")
      case _ =>
    }
    println("Enum vecs:")
    annos.foreach{
      case EnumVecAnnotation(target, enumTypeName, fields) => println(s"\t$target[$fields] => $enumTypeName")
      case _ =>
    }
  }
  // scalastyle:on regex

  def isCorrect(anno: EnumDefAnnotation, correct: CorrectDefAnno): Boolean = {
    (anno.typeName == correct.typeName ||
      anno.typeName.endsWith("." + correct.typeName) ||
      anno.typeName.endsWith("$" + correct.typeName)) &&
      anno.definition == correct.definition
  }

  def isCorrect(anno: EnumComponentAnnotation, correct: CorrectCompAnno): Boolean = {
    (anno.target match {
      case ComponentName(name, _) => name == correct.targetName
      case _ => throw new Exception("Unknown target type in EnumComponentAnnotation")
    }) &&
      (anno.enumTypeName == correct.typeName || anno.enumTypeName.endsWith("." + correct.typeName) ||
        anno.enumTypeName.endsWith("$" + correct.typeName))
  }

  def isCorrect(anno: EnumVecAnnotation, correct: CorrectVecAnno): Boolean = {
    (anno.target match {
      case ComponentName(name, _) => name == correct.targetName
      case _ => throw new Exception("Unknown target type in EnumVecAnnotation")
    }) &&
      (anno.typeName == correct.typeName || anno.typeName.endsWith("." + correct.typeName) ||
        anno.typeName.endsWith("$" + correct.typeName)) &&
      anno.fields.map(_.toSeq).toSet == correct.fields
  }

  def allCorrectDefs(annos: Seq[EnumDefAnnotation], corrects: Seq[CorrectDefAnno]): Boolean = {
    corrects.forall(c => annos.exists(isCorrect(_, c))) &&
      correctDefAnnos.length == annos.length
  }

  // Because temporary variables might be formed and annotated, we do not check that every component or vector
  // annotation is accounted for in the correct results listed above
  def allCorrectComps(annos: Seq[EnumComponentAnnotation], corrects: Seq[CorrectCompAnno]): Boolean =
    corrects.forall(c => annos.exists(isCorrect(_, c)))

  def allCorrectVecs(annos: Seq[EnumVecAnnotation], corrects: Seq[CorrectVecAnno]): Boolean =
    corrects.forall(c => annos.exists(isCorrect(_, c)))

  def test(strongEnumAnnotatorGen: () => Module) {
    val annos = (new ChiselStage).execute(Array("--target-dir", "test_run_dir", "--no-run-firrtl"),
                                          Seq(ChiselGeneratorAnnotation(strongEnumAnnotatorGen)))

    val enumDefAnnos = annos.collect { case a: EnumDefAnnotation => a }
    val enumCompAnnos = annos.collect { case a: EnumComponentAnnotation => a }
    val enumVecAnnos = annos.collect { case a: EnumVecAnnotation => a }

    allCorrectDefs(enumDefAnnos, correctDefAnnos) should be(true)
    allCorrectComps(enumCompAnnos, correctCompAnnos) should be(true)
    allCorrectVecs(enumVecAnnos, correctVecAnnos) should be(true)

  }

  "Test that strong enums annotate themselves appropriately" in {
    test(() => new StrongEnumAnnotator)
    test(() => new StrongEnumAnnotatorWithChiselName)
  }
}
