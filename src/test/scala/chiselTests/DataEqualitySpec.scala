package chiselTests

import chisel3._
import chisel3.experimental.VecLiterals._
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.Analog
import circt.stage.ChiselStage
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.Exceptions.AssertionFailed
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.Valid
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EqualityModule(lhsGen: => Data, rhsGen: => Data) extends Module {
  val out = IO(Output(Bool()))

  val lhs = lhsGen
  val rhs = rhsGen

  out := lhs === rhs
}

class EqualityTester(lhsGen: => Data, rhsGen: => Data) extends Module {
  val equalityModule = Module(new EqualityModule(lhsGen, rhsGen))

  assert(equalityModule.out)

  when(RegNext(next = true.B, init = false.B)) {
    stop()
  }
}

class AnalogBundle extends Bundle {
  val analog = Analog(32.W)
}

class AnalogExceptionModule extends Module {
  class AnalogExceptionModuleIO extends Bundle {
    val bundle1 = new AnalogBundle
    val bundle2 = new AnalogBundle
  }

  val io = IO(new AnalogExceptionModuleIO)
}

class AnalogExceptionTester extends Module {
  val module = Module(new AnalogExceptionModule)

  module.io.bundle1 <> DontCare
  module.io.bundle2 <> DontCare

  assert(module.io.bundle1 === module.io.bundle2)

  stop()
}

class DataEqualitySpec extends AnyFlatSpec with Matchers with ChiselSim {
  object MyEnum extends ChiselEnum {
    val sA, sB = Value
  }
  object MyEnumB extends ChiselEnum {
    val sA, sB = Value
  }
  class MyBundle extends Bundle {
    val a = UInt(8.W)
    val b = Bool()
    val c = MyEnum()
  }
  class LongBundle extends Bundle {
    val a = UInt(48.W)
    val b = SInt(32.W)
    val c = SInt(32.W)
  }
  class RuntimeSensitiveBundle(gen: => Bundle) extends Bundle {
    val a = UInt(8.W)
    val b: Bundle = gen
  }
  class MaybeEmptyBundle(x: Boolean) extends Bundle {
    val a = Option.when(x)(UInt(8.W))
  }

  behavior.of("UInt === UInt")
  it should "pass with equal values" in {
    simulate(new EqualityTester(0.U, 0.U))(RunUntilFinished(3))
  }
  it should "fail with differing values" in {
    intercept[AssertionFailed] { simulate(new EqualityTester(0.U, 1.U))(RunUntilFinished(3)) }
  }

  behavior.of("SInt === SInt")
  it should "pass with equal values" in {
    simulate(new EqualityTester(0.S, 0.S))(RunUntilFinished(3))
  }
  it should "fail with differing values" in {
    intercept[AssertionFailed] {
      simulate(
        new EqualityTester(0.S, 1.S)
      )(RunUntilFinished(3))
    }
  }

  behavior.of("Reset === Reset")
  it should "pass with equal values" in {
    simulate(
      new EqualityTester(true.B, true.B)
    )(RunUntilFinished(3))
  }
  it should "fail with differing values" in {
    intercept[AssertionFailed] {
      simulate(
        new EqualityTester(true.B, false.B)
      )(RunUntilFinished(3))
    }
  }
  it should "support abstract reset wires" in {
    simulate(
      new EqualityTester(WireDefault(Reset(), true.B), WireDefault(Reset(), true.B))
    )(RunUntilFinished(3))
  }

  behavior.of("AsyncReset === AsyncReset")
  it should "pass with equal values" in {
    simulate(
      new EqualityTester(true.B.asAsyncReset, true.B.asAsyncReset)
    )(RunUntilFinished(3))
  }
  it should "fail with differing values" in {
    intercept[AssertionFailed] {
      simulate(
        new EqualityTester(true.B.asAsyncReset, false.B.asAsyncReset)
      )(RunUntilFinished(3))
    }
  }

  behavior.of("ChiselEnum === ChiselEnum")
  it should "pass with equal values" in {
    simulate(
      new EqualityTester(MyEnum.sA, MyEnum.sA)
    )(RunUntilFinished(3))
  }
  it should "fail with differing values" in {
    intercept[AssertionFailed] {
      simulate(
        new EqualityTester(MyEnum.sA, MyEnum.sB)
      )(RunUntilFinished(3))
    }
  }

  behavior.of("Vec === Vec")
  it should "pass with equal sizes, equal values" in {
    simulate(
      new EqualityTester(
        Vec(3, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U),
        Vec(3, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U)
      )
    )(RunUntilFinished(3))
  }
  it should "support empty Vecs" in {
    simulate(
      new EqualityTester(
        Wire(Vec(0, UInt(8.W))),
        Wire(Vec(0, UInt(8.W)))
      )
    )(RunUntilFinished(3))
  }
  it should "fail with equal sizes, differing values" in {
    intercept[AssertionFailed] {
      simulate(
        new EqualityTester(
          Vec(3, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U),
          Vec(3, UInt(8.W)).Lit(0 -> 0.U, 1 -> 1.U, 2 -> 2.U)
        )
      )(RunUntilFinished(3))
    }
  }
  it should "throw a ChiselException with differing sizes" in {
    intercept[ChiselException] {
      ChiselStage.elaborate(
        new EqualityTester(
          Vec(3, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U),
          Vec(4, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U, 3 -> 4.U)
        )
      )
    }.getMessage should include("Vec sizes differ")
  }

  behavior.of("Bundle === Bundle")
  it should "pass with equal type, equal values" in {
    simulate(
      new EqualityTester(
        (new MyBundle).Lit(_.a -> 42.U, _.b -> false.B, _.c -> MyEnum.sB),
        (new MyBundle).Lit(_.a -> 42.U, _.b -> false.B, _.c -> MyEnum.sB)
      )
    )(RunUntilFinished(3))
  }
  it should "support empty Bundles" in {
    simulate(
      new EqualityTester(
        (new MaybeEmptyBundle(false)).Lit(),
        (new MaybeEmptyBundle(false)).Lit()
      )
    )(RunUntilFinished(3))
  }
  it should "fail with equal type, differing values" in {
    intercept[AssertionFailed] {
      simulate(
        new EqualityTester(
          (new MyBundle).Lit(_.a -> 42.U, _.b -> false.B, _.c -> MyEnum.sB),
          (new MyBundle).Lit(_.a -> 42.U, _.b -> false.B, _.c -> MyEnum.sA)
        )
      )(RunUntilFinished(3))
    }
  }
  it should "throw a ChiselException with differing runtime types" in {
    intercept[ChiselException] {
      ChiselStage.elaborate(
        new EqualityTester(
          (new RuntimeSensitiveBundle(new MyBundle)).Lit(
            _.a -> 1.U,
            _.b -> (new MyBundle).Lit(
              _.a -> 42.U,
              _.b -> false.B,
              _.c -> MyEnum.sB
            )
          ),
          (new RuntimeSensitiveBundle(new LongBundle)).Lit(
            _.a -> 1.U,
            _.b -> (new LongBundle).Lit(
              _.a -> 42.U,
              _.b -> 0.S,
              _.c -> 5.S
            )
          )
        )
      )
    }.getMessage should include("Runtime types differ")
  }

  behavior.of("DontCare === DontCare")
  it should "pass with two invalids" in {
    simulate(
      new EqualityTester(Valid(UInt(8.W)).Lit(_.bits -> 123.U), Valid(UInt(8.W)).Lit(_.bits -> 123.U))
    )(RunUntilFinished(3))
  }
  it should "exhibit the same behavior as comparing two invalidated wires" in {
    // Also check that two invalidated wires are equal
    simulate(
      new EqualityTester(WireInit(UInt(8.W), DontCare), WireInit(UInt(8.W), DontCare))
    )(RunUntilFinished(3))

    // Compare the verilog generated from both test cases and verify that they both are equal to true
    val verilog1 = ChiselStage.emitSystemVerilog(
      new EqualityModule(Valid(UInt(8.W)).Lit(_.bits -> 123.U), Valid(UInt(8.W)).Lit(_.bits -> 123.U))
    )
    val verilog2 =
      ChiselStage.emitSystemVerilog(new EqualityModule(WireInit(UInt(8.W), DontCare), WireInit(UInt(8.W), DontCare)))

    verilog1 should include("assign out = 1'h1;")
    verilog2 should include("assign out = 1'h1;")
  }

  behavior.of("Analog === Analog")
  it should "throw a ChiselException" in {
    intercept[ChiselException] { ChiselStage.elaborate(new AnalogExceptionTester) }.getMessage should include(
      "Equality isn't defined for Analog values"
    )
  }
}
