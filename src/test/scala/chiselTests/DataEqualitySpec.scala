package chiselTests

import chisel3._
import chisel3.experimental.VecLiterals._
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.Analog
import circt.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.util.Valid

class EqualityModule(lhsGen: => Data, rhsGen: => Data) extends Module {
  val out = IO(Output(Bool()))

  val lhs = lhsGen
  val rhs = rhsGen

  out := lhs === rhs
}

class EqualityTester(lhsGen: => Data, rhsGen: => Data) extends BasicTester {
  val equalityModule = Module(new EqualityModule(lhsGen, rhsGen))

  assert(equalityModule.out)

  stop()
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

class AnalogExceptionTester extends BasicTester {
  val module = Module(new AnalogExceptionModule)

  module.io.bundle1 <> DontCare
  module.io.bundle2 <> DontCare

  assert(module.io.bundle1 === module.io.bundle2)

  stop()
}

class DataEqualitySpec extends ChiselFlatSpec with Utils {
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

  behavior.of("UInt === UInt")
  it should "pass with equal values" in {
    assertTesterPasses {
      new EqualityTester(0.U, 0.U)
    }
  }
  it should "fail with differing values" in {
    assertTesterFails {
      new EqualityTester(0.U, 1.U)
    }
  }

  behavior.of("SInt === SInt")
  it should "pass with equal values" in {
    assertTesterPasses {
      new EqualityTester(0.S, 0.S)
    }
  }
  it should "fail with differing values" in {
    assertTesterFails {
      new EqualityTester(0.S, 1.S)
    }
  }

  behavior.of("Reset === Reset")
  it should "pass with equal values" in {
    assertTesterPasses {
      new EqualityTester(true.B, true.B)
    }
  }
  it should "fail with differing values" in {
    assertTesterFails {
      new EqualityTester(true.B, false.B)
    }
  }

  behavior.of("AsyncReset === AsyncReset")
  it should "pass with equal values" in {
    assertTesterPasses {
      new EqualityTester(true.B.asAsyncReset, true.B.asAsyncReset)
    }
  }
  it should "fail with differing values" in {
    assertTesterFails {
      new EqualityTester(true.B.asAsyncReset, false.B.asAsyncReset)
    }
  }

  behavior.of("ChiselEnum === ChiselEnum")
  it should "pass with equal values" in {
    assertTesterPasses {
      new EqualityTester(MyEnum.sA, MyEnum.sA)
    }
  }
  it should "fail with differing values" in {
    assertTesterFails {
      new EqualityTester(MyEnum.sA, MyEnum.sB)
    }
  }

  behavior.of("Vec === Vec")
  it should "pass with equal sizes, equal values" in {
    assertTesterPasses {
      new EqualityTester(
        Vec(3, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U),
        Vec(3, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U)
      )
    }
  }
  it should "fail with equal sizes, differing values" in {
    assertTesterFails {
      new EqualityTester(
        Vec(3, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U),
        Vec(3, UInt(8.W)).Lit(0 -> 0.U, 1 -> 1.U, 2 -> 2.U)
      )
    }
  }
  it should "throw a ChiselException with differing sizes" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      assertTesterFails {
        new EqualityTester(
          Vec(3, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U),
          Vec(4, UInt(8.W)).Lit(0 -> 1.U, 1 -> 2.U, 2 -> 3.U, 3 -> 4.U)
        )
      }
    }).getMessage should include("Vec sizes differ")
  }

  behavior.of("Bundle === Bundle")
  it should "pass with equal type, equal values" in {
    assertTesterPasses {
      new EqualityTester(
        (new MyBundle).Lit(_.a -> 42.U, _.b -> false.B, _.c -> MyEnum.sB),
        (new MyBundle).Lit(_.a -> 42.U, _.b -> false.B, _.c -> MyEnum.sB)
      )
    }
  }
  it should "fail with equal type, differing values" in {
    assertTesterFails {
      new EqualityTester(
        (new MyBundle).Lit(_.a -> 42.U, _.b -> false.B, _.c -> MyEnum.sB),
        (new MyBundle).Lit(_.a -> 42.U, _.b -> false.B, _.c -> MyEnum.sA)
      )
    }
  }
  it should "throw a ChiselException with differing runtime types" in {
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      assertTesterFails {
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
      }
    }).getMessage should include("Runtime types differ")
  }

  behavior.of("DontCare === DontCare")
  it should "pass with two invalids" in {
    assertTesterPasses {
      new EqualityTester(Valid(UInt(8.W)).Lit(_.bits -> 123.U), Valid(UInt(8.W)).Lit(_.bits -> 123.U))
    }
  }
  it should "exhibit the same behavior as comparing two invalidated wires" in {
    // Also check that two invalidated wires are equal
    assertTesterPasses {
      new EqualityTester(WireInit(UInt(8.W), DontCare), WireInit(UInt(8.W), DontCare))
    }

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
    (the[ChiselException] thrownBy extractCause[ChiselException] {
      assertTesterFails { new AnalogExceptionTester }
    }).getMessage should include("Equality isn't defined for Analog values")
  }
}
