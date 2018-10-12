// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.ChiselEnum
import chisel3.internal.firrtl.UnknownWidth
import chisel3.util._
import chisel3.testers.BasicTester
import org.scalatest.{FreeSpec, Matchers}

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
    val wire = WireInit(e)

    assert(e.isLit())
    assert(!wire.isLit())
  }
  stop()
}

class NextTester extends BasicTester {
  for ((e,n) <- EnumExample.all.zip(EnumExample.litValues.tail :+ EnumExample.litValues.head)) {
    assert(e.next.litValue == n.litValue)
    val w = WireInit(e)
    assert(w.next === EnumExample(n))
  }
  stop()
}

class WidthTester extends BasicTester {
  assert(EnumExample.getWidth == EnumExample.litValues.last.getWidth)
  assert(EnumExample.all.forall(_.getWidth == EnumExample.litValues.last.getWidth))
  assert(EnumExample.all.forall{e =>
    val w = WireInit(e)
    w.getWidth == EnumExample.litValues.last.getWidth
  })
  stop()
}

class StrongEnumFSMTester extends BasicTester {
  import StrongEnumFSM.State
  import StrongEnumFSM.State._

  val dut = Module(new StrongEnumFSM)

  // Inputs and expected results
  val inputs: Vec[Bool] = VecInit(false.B, true.B, false.B, true.B, true.B, true.B, false.B, true.B, true.B, false.B)
  val expected: Vec[Bool] = VecInit(false.B, false.B, false.B, false.B, false.B, true.B, true.B, false.B, false.B, true.B)
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

class StrongEnumSpec extends ChiselFlatSpec {
  import chisel3.internal.ChiselException

  behavior of "Strong enum tester"

  it should "fail to instantiate non-literal enums with the Value function" in {
    an [ExceptionInInitializerError] should be thrownBy {
      elaborate(new SimpleConnector(NonLiteralEnumType(), NonLiteralEnumType()))
    }
  }

  it should "fail to instantiate non-increasing enums with the Value function" in {
    an [ExceptionInInitializerError] should be thrownBy {
      elaborate(new SimpleConnector(NonIncreasingEnum(), NonIncreasingEnum()))
    }
  }

  it should "connect enums of the same type" in {
    elaborate(new SimpleConnector(EnumExample(), EnumExample()))
    elaborate(new SimpleConnector(EnumExample(), EnumExample.Type()))
  }

  it should "fail to connect a strong enum to a UInt" in {
    a [ChiselException] should be thrownBy {
      elaborate(new SimpleConnector(EnumExample(), UInt()))
    }
  }

  it should "fail to connect enums of different types" in {
    a [ChiselException] should be thrownBy {
      elaborate(new SimpleConnector(EnumExample(), OtherEnum()))
    }

    a [ChiselException] should be thrownBy {
      elaborate(new SimpleConnector(EnumExample.Type(), OtherEnum.Type()))
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
    a [ChiselException] should be thrownBy {
      elaborate(new CastToInvalidEnumTester)
    }
  }

  it should "only allow non-literal casts to enums if the width is smaller than or equal to the enum width" in {
    for (w <- 0 to EnumExample.getWidth)
      elaborate(new CastFromNonLitWidth(Some(w)))

    a [ChiselException] should be thrownBy {
      elaborate(new CastFromNonLitWidth)
    }

    for (w <- (EnumExample.getWidth+1) to (EnumExample.getWidth+100)) {
      a [ChiselException] should be thrownBy {
        elaborate(new CastFromNonLitWidth(Some(w)))
      }
    }
  }

  it should "execute enum comparison operations correctly" in {
    assertTesterPasses(new EnumOpsTester)
  }

  it should "fail to compare enums of different types" in {
    a [ChiselException] should be thrownBy {
      elaborate(new InvalidEnumOpsTester)
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
    def foo(e: EnumExample.Type) = {}

    "foo(EnumExample.e1); foo(EnumExample.e1.next)" should compile
    "foo(OtherEnum.otherEnum)" shouldNot compile
  }

  "StrongEnum FSM" should "work" in {
    assertTesterPasses(new StrongEnumFSMTester)
  }
}

class StrongEnumAnnotationSpec extends FreeSpec with Matchers {
  import chisel3.experimental.EnumAnnotations._
  import firrtl.annotations.ComponentName

  "Test that strong enums annotate themselves appropriately" in {

    def test() = {
      Driver.execute(Array("--target-dir", "test_run_dir"), () => new StrongEnumFSM) match {
        case ChiselExecutionSuccess(Some(circuit), emitted, _) =>
          val annos = circuit.annotations.map(_.toFirrtl)

          val enumDefAnnos = annos.collect { case a: EnumDefAnnotation => a }
          val enumCompAnnos = annos.collect { case a: EnumComponentAnnotation => a }

          // Print the annotations out onto the screen
          println("Enum definitions:")
          enumDefAnnos.foreach {
            case EnumDefAnnotation(enumTypeName, definition) => println(s"\t$enumTypeName: $definition")
          }
          println("Enum components:")
          enumCompAnnos.foreach{
            case EnumComponentAnnotation(target, enumTypeName) => println(s"\t$target => $enumTypeName")
          }

          // Check that the global annotation is correct
          enumDefAnnos.exists {
            case EnumDefAnnotation(name, map) =>
              name.endsWith("State") &&
                map.size == StrongEnumFSM.State.correct_annotation_map.size &&
                map.forall {
                  case (k, v) =>
                    val correctValue = StrongEnumFSM.State.correct_annotation_map(k)
                    correctValue == v
                }
            case _ => false
          } should be(true)

          // Check that the component annotations are correct
          enumCompAnnos.count {
            case EnumComponentAnnotation(target, enumName) =>
              val ComponentName(targetName, _) = target
              (targetName == "state" && enumName.endsWith("State")) ||
                (targetName == "io.state" && enumName.endsWith("State"))
            case _ => false
          } should be(2)

        case _ =>
          assert(false)
      }
    }

    // We run this test twice, to test for an older bug where only the first circuit would be annotated
    test()
    test()
  }
}
