// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.AffectsChiselPrefix
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.ChiselStage
import chisel3.util.{is, switch, Cat, Counter}
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.testing.scalatest.FileCheck
import java.io.ByteArrayOutputStream
import org.scalatest.Assertion
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.freespec.AnyFreeSpec
import org.scalatest.matchers.should.Matchers
import scala.Console.{withErr, withOut}
import scala.annotation.nowarn

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

  io.out := io.in.asUInt
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

class SafeCastFromNonLit extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(EnumExample.getWidth.W))
    val out = Output(EnumExample())
    val valid = Output(Bool())
  })

  val (enumVal, valid) = EnumExample.safe(io.in)
  io.out := enumVal
  io.valid := valid
}

class CastFromNonLitWidth(w: Option[Int] = None) extends Module {
  val width = if (w.isDefined) w.get.W else UnknownWidth

  val io = IO(new Bundle {
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

object ChiselEnumFSM {
  object State extends ChiselEnum {
    val sNone, sOne1, sTwo1s = Value

    val correct_annotation_map = Map[String, BigInt]("sNone" -> 0, "sOne1" -> 1, "sTwo1s" -> 2)
  }
}

class ChiselEnumFSM extends Module {
  import ChiselEnumFSM.State
  import ChiselEnumFSM.State._

  // This FSM detects two 1's one after the other
  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
    val state = Output(State())
  })

  val state = RegInit(sNone)

  io.out := (state === sTwo1s)
  io.state := state

  switch(state) {
    is(sNone) {
      when(io.in) {
        state := sOne1
      }
    }
    is(sOne1) {
      when(io.in) {
        state := sTwo1s
      }.otherwise {
        state := sNone
      }
    }
    is(sTwo1s) {
      when(!io.in) {
        state := sNone
      }
    }
  }
}

object Opcode extends ChiselEnum {
  val load = Value(0x03.U)
  val imm = Value(0x13.U)
  val auipc = Value(0x17.U)
  val store = Value(0x23.U)
  val reg = Value(0x33.U)
  val lui = Value(0x37.U)
  val br = Value(0x63.U)
  val jalr = Value(0x67.U)
  val jal = Value(0x6f.U)
}

class LoadStoreExample extends Module {
  val io = IO(new Bundle {
    val opcode = Input(Opcode())
    val load_or_store = Output(Bool())
  })
  io.load_or_store := io.opcode.isOneOf(Opcode.load, Opcode.store)
  printf(p"${io.opcode}")
}

class CastToUIntTester extends Module {
  for ((enumVal, lit) <- EnumExample.all.zip(EnumExample.litValues)) {
    val mod = Module(new CastToUInt)
    mod.io.in := enumVal
    assert(mod.io.out === lit)
  }
  stop()
}

class CastFromLitTester extends Module {
  for ((enumVal, lit) <- EnumExample.all.zip(EnumExample.litValues)) {
    val mod = Module(new CastFromLit(lit))
    assert(mod.io.out === enumVal)
    assert(mod.io.valid === true.B)
  }
  stop()
}

class CastFromNonLitTester extends Module {
  for ((enumVal, lit) <- EnumExample.all.zip(EnumExample.litValues)) {
    val mod = Module(new CastFromNonLit)
    mod.io.in := lit
    assert(mod.io.out === enumVal)
    assert(mod.io.valid === true.B)
  }

  val invalid_values =
    (1 until (1 << EnumExample.getWidth)).filter(!EnumExample.litValues.map(_.litValue.toInt).contains(_)).map(_.U)

  for (invalid_val <- invalid_values) {
    val mod = Module(new CastFromNonLit)
    mod.io.in := invalid_val

    assert(mod.io.valid === false.B)
  }

  stop()
}

class SafeCastFromNonLitTester extends Module {
  for ((enumVal, lit) <- EnumExample.all.zip(EnumExample.litValues)) {
    val mod = Module(new SafeCastFromNonLit)
    mod.io.in := lit
    assert(mod.io.out === enumVal)
    assert(mod.io.valid === true.B)
  }

  val invalid_values =
    (1 until (1 << EnumExample.getWidth)).filter(!EnumExample.litValues.map(_.litValue.toInt).contains(_)).map(_.U)

  for (invalid_val <- invalid_values) {
    val mod = Module(new SafeCastFromNonLit)
    mod.io.in := invalid_val

    assert(mod.io.valid === false.B)
  }

  stop()
}

class CastToInvalidEnumTester extends Module {
  val invalid_value: UInt = EnumExample.litValues.last + 1.U
  Module(new CastFromLit(invalid_value))
}

class EnumOpsTester extends Module {
  for {
    x <- EnumExample.all
    y <- EnumExample.all
  } {
    val mod = Module(new EnumOps(EnumExample, EnumExample))
    mod.io.x := x
    mod.io.y := y

    assert(mod.io.lt === (x.asUInt < y.asUInt))
    assert(mod.io.le === (x.asUInt <= y.asUInt))
    assert(mod.io.gt === (x.asUInt > y.asUInt))
    assert(mod.io.ge === (x.asUInt >= y.asUInt))
    assert(mod.io.eq === (x.asUInt === y.asUInt))
    assert(mod.io.ne === (x.asUInt =/= y.asUInt))
  }
  stop()
}

class InvalidEnumOpsTester extends Module {
  val mod = Module(new EnumOps(EnumExample, OtherEnum))
  mod.io.x := EnumExample.e0
  mod.io.y := OtherEnum.otherEnum
}

class IsLitTester extends Module {
  for (e <- EnumExample.all) {
    val wire = WireDefault(e)

    assert(e.isLit)
    assert(!wire.isLit)
  }
  stop()
}

class NextTester extends Module {
  for ((e, n) <- EnumExample.all.zip(EnumExample.litValues.tail :+ EnumExample.litValues.head)) {
    assert(e.next.litValue == n.litValue)
    val w = WireDefault(e)
    assert(w.next === EnumExample(n))
  }
  stop()
}

class WidthTester extends Module {
  assert(EnumExample.getWidth == EnumExample.litValues.last.getWidth)
  assert(EnumExample.all.forall(_.getWidth == EnumExample.litValues.last.getWidth))
  assert(EnumExample.all.forall { e =>
    val w = WireDefault(e)
    w.getWidth == EnumExample.litValues.last.getWidth
  })
  stop()
}

class KnownWidthTester extends Module {
  object EnumWithKnownWidth extends ChiselEnum {
    val Member0 = Value(0.U(1.W))
    val Member1 = Value(1.U(2.W))
    val Member2 = Value(2.U(3.W))
    val Member3 = Value(3.U(4.W))
  }
  assert(EnumWithKnownWidth.getWidth == 4)
  assert(EnumWithKnownWidth.all.forall(_.getWidth == 4))
  stop()
}

class ChiselEnumFSMTester extends Module {
  import ChiselEnumFSM.State._

  val dut = Module(new ChiselEnumFSM)

  // Inputs and expected results
  val inputs: Vec[Bool] = VecInit(false.B, true.B, false.B, true.B, true.B, true.B, false.B, true.B, true.B, false.B)
  val expected: Vec[Bool] =
    VecInit(false.B, false.B, false.B, false.B, false.B, true.B, true.B, false.B, false.B, true.B)
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

class IsOneOfTester extends Module {
  import EnumExample._

  // is one of itself
  assert(e0.isOneOf(e0))

  // is one of Seq of itself
  assert(e0.isOneOf(Seq(e0)))
  assert(e0.isOneOf(Seq(e0, e0, e0, e0)))
  assert(e0.isOneOf(e0, e0, e0, e0))

  // is one of Seq of multiple elements
  val subset = Seq(e0, e1, e2)
  assert(e0.isOneOf(subset))
  assert(e1.isOneOf(subset))
  assert(e2.isOneOf(subset))

  // is not element not in subset
  assert(!e100.isOneOf(subset))
  assert(!e101.isOneOf(subset))

  // test multiple elements with variable number of arguments
  assert(e0.isOneOf(e0, e1, e2))
  assert(e1.isOneOf(e0, e1, e2))
  assert(e2.isOneOf(e0, e1, e2))
  assert(!e100.isOneOf(e0, e1, e2))
  assert(!e101.isOneOf(e0, e1, e2))

  // is not another value
  assert(!e0.isOneOf(e1))
  assert(!e2.isOneOf(e101))

  stop()
}

class ChiselEnumSpec extends AnyFlatSpec with Matchers with LogUtils with ChiselSim with FileCheck {

  behavior.of("ChiselEnum")

  it should "fail to instantiate non-literal enums with the Value function" in {
    intercept[ExceptionInInitializerError] {
      ChiselStage.emitCHIRRTL(new SimpleConnector(NonLiteralEnumType(), NonLiteralEnumType()))
    }
  }

  it should "fail to instantiate non-increasing enums with the Value function" in {
    intercept[ExceptionInInitializerError] {
      ChiselStage.emitCHIRRTL(new SimpleConnector(NonIncreasingEnum(), NonIncreasingEnum()))
    }
  }

  it should "connect enums of the same type" in {
    ChiselStage.emitCHIRRTL(new SimpleConnector(EnumExample(), EnumExample()))
    ChiselStage.emitCHIRRTL(new SimpleConnector(EnumExample(), EnumExample.Type()))
  }

  it should "fail to connect a strong enum to a UInt" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new SimpleConnector(EnumExample(), UInt()))
    }
  }

  it should "fail to connect enums of different types" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new SimpleConnector(EnumExample(), OtherEnum()))
    }

    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new SimpleConnector(EnumExample.Type(), OtherEnum.Type()))
    }
  }

  it should "cast enums to UInts correctly" in {
    simulate(new CastToUIntTester)(RunUntilFinished(3))
  }

  // See: https://github.com/chipsalliance/chisel/issues/4159
  it should "give the correct width for Chisel Enum values" in {
    val verilog = ChiselStage.emitSystemVerilog(new RawModule {
      val out1, out2, out3 = IO(Output(UInt(8.W)))
      val e = EnumExample.e1
      val x = e.asUInt
      val y = e.asTypeOf(UInt())
      val z = e.asTypeOf(UInt(e.getWidth.W))
      out1 := Cat(1.U, x)
      out2 := Cat(1.U, y)
      out3 := Cat(1.U, z)
      x.getWidth should be(7)
      x.getWidth should be(EnumExample.getWidth)
      y.getWidth should be(7)
      z.getWidth should be(7)
    })
    verilog should include("assign out1 = 8'h81;")
    verilog should include("assign out2 = 8'h81;")
    verilog should include("assign out3 = 8'h81;")
  }

  // This is a bug, but fixing it may break user code.
  // See: https://github.com/chipsalliance/chisel/issues/4159
  it should "preserve legacy width behavior with --use-legacy-width" in {
    val verilog = ChiselStage.emitSystemVerilog(
      new RawModule {
        val out1, out2, out3 = IO(Output(UInt(8.W)))
        val e = EnumExample.e1
        val x = e.asUInt
        val y = e.asTypeOf(UInt())
        val z = e.asTypeOf(UInt(e.getWidth.W))
        out1 := Cat(1.U, x)
        out2 := Cat(1.U, y)
        out3 := Cat(1.U, z)
        // The bug is that the width of x is 7 but the value of out1 is 3
        x.getWidth should be(7)
        x.getWidth should be(EnumExample.getWidth)
        y.getWidth should be(7)
        z.getWidth should be(7)
      },
      args = Array("--use-legacy-width")
    )
    // The bug is that all of these should be the same as out3, or the widths above are wrong
    verilog should include("assign out1 = 8'h3;")
    verilog should include("assign out2 = 8'h3;")
    verilog should include("assign out3 = 8'h81;")
  }

  it should "cast literal UInts to enums correctly" in {
    simulate(new CastFromLitTester)(RunUntilFinished(3))
  }

  it should "cast non-literal UInts to enums correctly and detect illegal casts" in {
    simulate(new CastFromNonLitTester)(RunUntilFinished(3))
  }

  it should "safely cast non-literal UInts to enums correctly and detect illegal casts" in {
    simulate(new SafeCastFromNonLitTester)(RunUntilFinished(3))
  }

  it should "prevent illegal literal casts to enums" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new CastToInvalidEnumTester)
    }
  }

  it should "only allow non-literal casts to enums if the width is smaller than or equal to the enum width" in {
    for (w <- 0 to EnumExample.getWidth)
      ChiselStage.emitCHIRRTL(new CastFromNonLitWidth(Some(w)))

    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new CastFromNonLitWidth)
    }

    for (w <- (EnumExample.getWidth + 1) to (EnumExample.getWidth + 100)) {
      intercept[ChiselException] {
        ChiselStage.emitCHIRRTL(new CastFromNonLitWidth(Some(w)))
      }
    }
  }

  it should "execute enum comparison operations correctly" in {
    simulate(new EnumOpsTester)(RunUntilFinished(3))
  }

  it should "fail to compare enums of different types" in {
    intercept[ChiselException] {
      ChiselStage.emitCHIRRTL(new InvalidEnumOpsTester)
    }
  }

  it should "correctly check whether or not enums are literal" in {
    simulate(new IsLitTester)(RunUntilFinished(3))
  }

  it should "return the correct next values for enums" in {
    simulate(new NextTester)(RunUntilFinished(3))
  }

  it should "return the correct widths for enums" in {
    simulate(new WidthTester)(RunUntilFinished(3))
  }

  it should "return the correct widths for enums with known-width values" in {
    simulate(new KnownWidthTester)(RunUntilFinished(3))
  }

  it should "maintain Scala-level type-safety" in {
    def foo(e: EnumExample.Type): Unit = {}

    "foo(EnumExample.e1); foo(EnumExample.e1.next)" should compile
    "foo(OtherEnum.otherEnum)" shouldNot compile
  }

  it should "prevent enums from being declared without names" in {
    "object UnnamedEnum extends ChiselEnum { Value }" shouldNot compile
  }

  "ChiselEnum FSM" should "work" in {
    simulate(new ChiselEnumFSMTester)(RunUntilFinished(11))
  }

  "Casting a UInt to an Enum" should "warn if the UInt can express illegal states" in {
    object MyEnum extends ChiselEnum {
      val e0, e1, e2 = Value
    }

    class MyModule extends Module {
      val in = IO(Input(UInt(2.W)))
      val out = IO(Output(MyEnum()))
      out := MyEnum(in)
    }
    val outStream = new ByteArrayOutputStream()
    withOut(outStream)(withErr(outStream)(ChiselStage.emitCHIRRTL(new MyModule)))
    outStream.toString.fileCheck()(
      """| CHECK:      [W001] Casting non-literal UInt to [[enum:[a-zA-Z0-9_$.]+]].
         | CHECK-SAME: You can use [[enum]].safe to cast without this warning.
         |""".stripMargin
    )
  }

  it should "NOT warn if the Enum is total" in {
    object TotalEnum extends ChiselEnum {
      val e0, e1, e2, e3 = Value
    }

    class MyModule extends Module {
      val in = IO(Input(UInt(2.W)))
      val out = IO(Output(TotalEnum()))
      out := TotalEnum(in)
    }
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new MyModule))
    (log should not).include("warn")
  }

  it should "suppress warning using suppressEnumCastWarning" in {
    object TestEnum extends ChiselEnum {
      val e0, e1, e2 = Value
    }

    class MyModule extends Module {
      val in = IO(Input(UInt(2.W)))
      val out = IO(Output(TestEnum()))
      suppressEnumCastWarning {
        val res = TestEnum(in)
        out := res
      }
    }
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new MyModule))
    (log should not).include("warn")
  }

  it should "suppress exactly one warning using suppressEnumCastWarning" in {
    object TestEnum1 extends ChiselEnum {
      val e0, e1, e2 = Value
    }
    object TestEnum2 extends ChiselEnum {
      val e0, e1, e2 = Value
    }

    class MyModule extends Module {
      val in = IO(Input(UInt(2.W)))
      val out1 = IO(Output(TestEnum1()))
      val out2 = IO(Output(TestEnum2()))
      suppressEnumCastWarning {
        out1 := TestEnum1(in)
      }
      out2 := TestEnum2(in)
    }
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new MyModule))
    log should include("warn")
    log should include("TestEnum2") // not suppressed
    (log should not).include("TestEnum1") // suppressed
  }

  "Casting a UInt to an Enum with .safe" should "NOT warn" in {
    object MyEnum extends ChiselEnum {
      val e0, e1, e2 = Value
    }

    class MyModule extends Module {
      val in = IO(Input(UInt(2.W)))
      val out = IO(Output(MyEnum()))
      out := MyEnum.safe(in)._1
    }
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new MyModule))
    (log should not).include("warn")
  }

  it should "NOT generate any validity logic if the Enum is total" in {
    object TotalEnum extends ChiselEnum {
      val e0, e1, e2, e3 = Value
    }

    class MyModule extends Module {
      val in = IO(Input(UInt(2.W)))
      val out = IO(Output(TotalEnum()))
      val (res, valid) = TotalEnum.safe(in)
      assert(valid.litToBoolean, "It should be true.B")
      out := res
    }
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new MyModule))
    (log should not).include("warn")
  }

  it should "correctly check if the enumeration is one of the values in a given sequence" in {
    simulate(new IsOneOfTester)(RunUntilFinished(3))
  }

  it should "work with Printables" in {
    ChiselStage.emitCHIRRTL(new LoadStoreExample) should include(
      """printf(clock, UInt<1>(0h1), "%c%c%c%c%c", _chiselTestsOpcodePrintable[0], _chiselTestsOpcodePrintable[1], _chiselTestsOpcodePrintable[2], _chiselTestsOpcodePrintable[3], _chiselTestsOpcodePrintable[4])"""
    )
  }
}

class ChiselEnumAnnotator extends Module {
  import EnumExample._

  object LocalEnum extends ChiselEnum {
    val le0, le1 = Value
    val le2 = Value
    val le100 = Value(100.U)
  }

  val io = IO(new Bundle {
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

class ChiselEnumAnnotatorWithChiselName extends Module {
  import EnumExample._

  object LocalEnum extends ChiselEnum with AffectsChiselPrefix {
    val le0, le1 = Value
    val le2 = Value
    val le100 = Value(100.U)
  }

  val io = IO(new Bundle {
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
