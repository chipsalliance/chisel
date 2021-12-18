// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester

import org.scalacheck.Gen
import org.scalatestplus.scalacheck.ScalaCheckDrivenPropertyChecks

import scala.collection.immutable.ListMap

// Need separate import to override compile options from Chisel._
object CompatibilityCustomCompileOptions {
  import Chisel.{defaultCompileOptions => _, _}
  implicit val customCompileOptions =
    chisel3.ExplicitCompileOptions.NotStrict.copy(inferModuleReset = true)
  class Foo extends Module {
    val io = new Bundle {}
  }
}

class CompatibiltySpec extends ChiselFlatSpec with ScalaCheckDrivenPropertyChecks with Utils {
  import Chisel._

  behavior of "Chisel compatibility layer"

  it should "accept direction arguments" in {
    ChiselStage.elaborate(new Module {
      // Choose a random direction
      val directionArgument: Direction = Gen.oneOf(INPUT, OUTPUT, NODIR).sample.get
      val expectedDirection = directionArgument match {
        case NODIR => OUTPUT
        case other => other
      }
      // Choose a random width
      val width = Gen.choose(1, 2048).sample.get
      val io = new Bundle {
        val b = Bool(directionArgument)
        val u = UInt(directionArgument, width)
      }
      io.b shouldBe a [Bool]
      io.b.getWidth shouldEqual 1
      io.b.dir shouldEqual (expectedDirection)
      io.u shouldBe a [UInt]
      io.u.getWidth shouldEqual width
      io.u.dir shouldEqual (expectedDirection)
    })
  }

  it should "accept single argument U/SInt factory methods" in {
    // Choose a random value
    val value: Int = Gen.choose(0, Int.MaxValue).sample.get
    val l = UInt(value)
    l shouldBe a [UInt]
    l shouldBe 'lit
    l.getWidth shouldEqual BigInt(value).bitLength
    l.litValue() shouldEqual value
  }

  it should "map utility objects into the package object" in {
    val value: Int = Gen.choose(2, 2048).sample.get
    log2Up(value) shouldBe (1 max BigInt(value - 1).bitLength)
    log2Ceil(value) shouldBe (BigInt(value - 1).bitLength)
    log2Down(value) shouldBe ((1 max BigInt(value - 1).bitLength) - (if (value > 0 && ((value & (value - 1)) == 0)) 0 else 1))
    log2Floor(value) shouldBe (BigInt(value - 1).bitLength - (if (value > 0 && ((value & (value - 1)) == 0)) 0 else 1))
    isPow2(BigInt(1) << value) shouldBe true
    isPow2((BigInt(1) << value) - 1) shouldBe false
  }

  it should "make BitPats available" in {
    val value: Int = Gen.choose(1, Int.MaxValue).sample.get
    val binaryString = value.toBinaryString
    val maskPosition = Gen.choose(0, binaryString.length - 1).sample.get
    val bs = new StringBuilder(binaryString)
    bs(maskPosition) = '?'
    val bitPatString = bs.toString
    val bp = BitPat("b" + bitPatString)
    bp shouldBe a [BitPat]
    bp.getWidth shouldEqual binaryString.length

  }

  it should "successfully compile a complete module" in {
    class Dummy extends Module {
      // The following just checks that we can create objects with nothing more than the Chisel compatibility package.
      val io = new Bundle {}
      val data = UInt(width = 3)
      val wire = Wire(data)
      new ArbiterIO(data, 2) shouldBe a [ArbiterIO[UInt]]
      Module(new LockingRRArbiter(data, 2, 2, None)) shouldBe a [LockingRRArbiter[UInt]]
      Module(new RRArbiter(data, 2)) shouldBe a [RRArbiter[UInt]]
      Module(new Arbiter(data, 2)) shouldBe a [Arbiter[UInt]]
      new Counter(2) shouldBe a [Counter]
      new ValidIO(data) shouldBe a [ValidIO[UInt]]
      new DecoupledIO(data) shouldBe a [DecoupledIO[UInt]]
      new QueueIO(data, 2) shouldBe a [QueueIO[UInt]]
      Module(new Pipe(data, 2)) shouldBe a [Pipe[UInt]]

      FillInterleaved(2, wire) shouldBe a [UInt]
      PopCount(wire) shouldBe a [UInt]
      Fill(2, wire) shouldBe a [UInt]
      Reverse(wire) shouldBe a [UInt]
      Cat(wire, wire) shouldBe a [UInt]
      Log2(wire) shouldBe a [UInt]
      // 'switch' and 'is' are tested below in Risc
      Counter(2) shouldBe a [Counter]
      DecoupledIO(wire) shouldBe a [DecoupledIO[UInt]]
      val dcd = Wire(Decoupled(data))
      dcd shouldBe a [DecoupledIO[UInt]]
      Queue(dcd) shouldBe a [DecoupledIO[UInt]]
      Queue(dcd, 0) shouldBe a [DecoupledIO[UInt]]
      Enum(UInt(), 2) shouldBe a [List[UInt]]
      ListLookup(wire, List(wire), Array((BitPat("b1"), List(wire)))) shouldBe a [List[UInt]]
      Lookup(wire, wire, Seq((BitPat("b1"), wire))) shouldBe a [UInt]
      Mux1H(wire, Seq(wire)) shouldBe a [UInt]
      PriorityMux(Seq(Bool(false)), Seq(data)) shouldBe a [UInt]
      MuxLookup(wire, wire, Seq((wire, wire))) shouldBe a [UInt]
      MuxCase(wire, Seq((Bool(true), wire))) shouldBe a [UInt]
      OHToUInt(wire) shouldBe a [UInt]
      PriorityEncoder(wire) shouldBe a [UInt]
      UIntToOH(wire) shouldBe a [UInt]
      PriorityEncoderOH(wire) shouldBe a [UInt]
      RegNext(wire) shouldBe a [UInt]
      RegInit(wire) shouldBe a [UInt]
      RegEnable(wire, Bool(true)) shouldBe a [UInt]
      ShiftRegister(wire, 2) shouldBe a [UInt]
      Valid(data) shouldBe a [ValidIO[UInt]]
      Pipe(Wire(Valid(data)), 2) shouldBe a [ValidIO[UInt]]
    }
    ChiselStage.elaborate { new Dummy }
  }
  // Verify we can elaborate a design expressed in Chisel2
  class Chisel2CompatibleRisc extends Module {
    val io = new Bundle {
      val isWr   = Bool(INPUT)
      val wrAddr = UInt(INPUT, 8)
      val wrData = Bits(INPUT, 32)
      val boot   = Bool(INPUT)
      val valid  = Bool(OUTPUT)
      val out    = Bits(OUTPUT, 32)
    }
    val file = Mem(256, Bits(width = 32))
    val code = Mem(256, Bits(width = 32))
    val pc   = Reg(init=UInt(0, 8))

    val add_op :: imm_op :: Nil = Enum(2)

    val inst = code(pc)
    val op   = inst(31,24)
    val rci  = inst(23,16)
    val rai  = inst(15, 8)
    val rbi  = inst( 7, 0)

    val ra = Mux(rai === Bits(0), Bits(0), file(rai))
    val rb = Mux(rbi === Bits(0), Bits(0), file(rbi))
    val rc = Wire(Bits(width = 32))

    io.valid := Bool(false)
    io.out   := Bits(0)
    rc       := Bits(0)

    when (io.isWr) {
      code(io.wrAddr) := io.wrData
    } .elsewhen (io.boot) {
      pc := UInt(0)
    } .otherwise {
      switch(op) {
        is(add_op) { rc := ra +% rb }
        is(imm_op) { rc := (rai << 8) | rbi }
      }
      io.out := rc
      when (rci === UInt(255)) {
        io.valid := Bool(true)
      } .otherwise {
        file(rci) := rc
      }
      pc := pc +% UInt(1)
    }
  }

  it should "Chisel2CompatibleRisc should elaborate" in {
    ChiselStage.elaborate { new Chisel2CompatibleRisc }
  }

  it should "not try to assign directions to Analog" in {
    ChiselStage.elaborate(new Module {
      val io = new Bundle {
        val port = chisel3.experimental.Analog(32.W)
      }
    })
  }


  class SmallBundle extends Bundle {
    val f1 = UInt(width = 4)
    val f2 = UInt(width = 5)
  }
  class BigBundle extends SmallBundle {
    val f3 = UInt(width = 6)
  }

  "A Module with missing bundle fields when compiled with the Chisel compatibility package" should "not throw an exception" in {

    class ConnectFieldMismatchModule extends Module {
      val io = new Bundle {
        val in = (new SmallBundle).asInput
        val out = (new BigBundle).asOutput
      }
      io.out := io.in
    }
    ChiselStage.elaborate { new ConnectFieldMismatchModule() }
  }

  "A Module in which a Reg is created with a bound type when compiled with the Chisel compatibility package" should "not throw an exception" in {

    class CreateRegFromBoundTypeModule extends Module {
      val io = new Bundle {
        val in = (new SmallBundle).asInput
        val out = (new BigBundle).asOutput
      }
      val badReg = Reg(UInt(7, width=4))
    }
    ChiselStage.elaborate { new CreateRegFromBoundTypeModule() }
  }

  "A Module with unwrapped IO when compiled with the Chisel compatibility package" should "not throw an exception" in {

    class RequireIOWrapModule extends Module {
      val io = new Bundle {
        val in = UInt(width = 32).asInput
        val out = Bool().asOutput
      }
      io.out := io.in(1)
    }
    ChiselStage.elaborate { new RequireIOWrapModule() }
  }

  "A Module without val io" should "throw an exception" in {
    class ModuleWithoutValIO extends Module {
      val foo = new Bundle {
        val in = UInt(width = 32).asInput
        val out = Bool().asOutput
      }
      foo.out := foo.in(1)
    }
    val e = intercept[Exception](
      ChiselStage.elaborate { new ModuleWithoutValIO }
    )
    e.getMessage should include("must have a 'val io' Bundle")
  }

  "A Module connecting output as source to input as sink when compiled with the Chisel compatibility package" should "not throw an exception" in {

    class SimpleModule extends Module {
      val io = new Bundle {
        val in = (UInt(width = 3)).asInput
        val out = (UInt(width = 4)).asOutput
      }
    }
    class SwappedConnectionModule extends SimpleModule {
      val child = Module(new SimpleModule)
      io.in := child.io.out
    }
    ChiselStage.elaborate { new SwappedConnectionModule() }
  }

  "Vec ports" should "give default directions to children so they can be used in chisel3.util" in {
    import Chisel._
    ChiselStage.elaborate(new Module {
      val io = new Bundle {
        val in = Vec(1, UInt(width = 8)).flip
        val out = UInt(width = 8)
      }
      io.out := RegEnable(io.in(0), true.B)
    })
  }

  "Reset" should "still walk, talk, and quack like a Bool" in {
    import Chisel._
    ChiselStage.elaborate(new Module {
      val io = new Bundle {
        val in = Bool(INPUT)
        val out = Bool(OUTPUT)
      }
      io.out := io.in && reset
    })
  }

  "Data.dir" should "give the correct direction for io" in {
    import Chisel._
    ChiselStage.elaborate(new Module {
      val io = (new Bundle {
        val foo = Bool(OUTPUT)
        val bar = Bool().flip
      }).flip
      Chisel.assert(io.foo.dir == INPUT)
      Chisel.assert(io.bar.dir == OUTPUT)
    })
  }

  // Note: This is a regression (see https://github.com/freechipsproject/chisel3/issues/668)
  it should "fail for Chisel types" in {
    import Chisel._
    an [chisel3.ExpectedHardwareException] should be thrownBy extractCause[chisel3.ExpectedHardwareException] {
      ChiselStage.elaborate(new Module {
        val io = new Bundle { }
        UInt(INPUT).dir
      })
    }
  }

  "Mux return value" should "be able to be used on the RHS" in {
    import Chisel._
    ChiselStage.elaborate(new Module {
      val gen = new Bundle { val foo = UInt(width = 8) }
      val io = new Bundle {
        val a = Vec(2, UInt(width = 8)).asInput
        val b = Vec(2, UInt(width = 8)).asInput
        val c = gen.asInput
        val d = gen.asInput
        val en = Bool(INPUT)
        val y = Vec(2, UInt(width = 8)).asOutput
        val z = gen.asOutput
      }
      io.y := Mux(io.en, io.a, io.b)
      io.z := Mux(io.en, io.c, io.d)
    })
  }

  "Chisel3 IO constructs" should "be useable in Chisel2" in {
    import Chisel._
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val in = Input(Bool())
        val foo = Output(Bool())
        val bar = Flipped(Bool())
      })
      Chisel.assert(io.in.dir == INPUT)
      Chisel.assert(io.foo.dir == OUTPUT)
      Chisel.assert(io.bar.dir == INPUT)
    })
  }

  behavior of "BitPat"

  it should "support old operators" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      info("Deprecated method DC hasn't been removed")
      val bp = BitPat.DC(4)
    }

    ChiselStage.elaborate(new Foo)
  }

  behavior of "Enum"

  it should "support apply[T <: Bits](nodeType: T, n: Int): List[T]" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      info("works for a UInt")
      Enum(UInt(), 4) shouldBe a [List[UInt]]

      info("throw an exception for non-UInt types")
      intercept [IllegalArgumentException] {
        Enum(SInt(), 4)
      }.getMessage should include ("Only UInt supported for enums")

      info("throw an exception if the bit width is specified")
      intercept [IllegalArgumentException] {
        Enum(UInt(width = 8), 4)
      }.getMessage should include ("Bit width may no longer be specified for enums")
    }

    ChiselStage.elaborate(new Foo)
  }

  behavior of "Queue"

  it should "support deprecated constructors" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      info("reset: Option[Bool] constructor works")
      val option = Module(new Queue(UInt(), 4, false, false, Some(Bool(true))))

      info("reset: Bool constructor works")
      val explicit = Module(new Queue(UInt(), 4, false, false, Bool(true)))
    }

    ChiselStage.elaborate(new Foo)
  }

  behavior of "LFSR16"

  it should "still exist" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      info("Still exists")
      val lfsr = LFSR16()

      info("apply method returns a UInt")
      lfsr shouldBe a [UInt]

      info("returned UInt has a width of 16")
      lfsr.getWidth should be (16)
    }

    ChiselStage.elaborate(new Foo)
  }

  behavior of "Mem"

  it should "support deprecated apply methods" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      info("apply[T <: Data](t: T, size: BigInt): Mem[T] works")
      val memBigInt = Mem(UInt(), 8: BigInt)
      memBigInt shouldBe a [Mem[UInt]]

      info("apply[T <: Data](t: T, size: Int): Mem[T] works")
      val memInt = Mem(SInt(), 16: Int)
      memInt shouldBe a [Mem[SInt]]
    }

    ChiselStage.elaborate(new Foo)
  }

  behavior of "SeqMem"

  it should "support deprecated apply methods" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      info("apply[T <: Data](t: T, size: BigInt): SeqMem[T] works")
      val seqMemBigInt = SeqMem(UInt(), 8: BigInt)
      seqMemBigInt shouldBe a [SeqMem[UInt]]

      info("apply[T <: Data](t: T, size: Int): SeqMem[T] works")
      val seqMemInt = SeqMem(UInt(), 16: Int)
      seqMemInt shouldBe a [SeqMem[UInt]]
    }

    ChiselStage.elaborate(new Foo)
  }

  it should "support data-types of mixed directionality" in {
    class Foo extends Module {
      val io = IO(new Bundle {})
      val tpe = new Bundle { val foo = UInt(OUTPUT, width = 4); val bar = UInt(width = 4) }
      // NOTE for some reason, the old bug this hit did not occur when `tpe` is inlined
      val mem = SeqMem(tpe, 8)
      mem(3.U)

    }
    ChiselStage.elaborate((new Foo))
  }

  behavior of "debug"

  it should "still exist" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      val data = UInt(width = 2)
      debug(data)
    }

    ChiselStage.elaborate(new Foo)
  }

  behavior of "Data methods"

  behavior of "Wire"

  it should "support legacy methods" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      info("apply[T <: Data](dummy: Int = 0, init: T): T works")
      val first = Wire(init=UInt("hdeadbeef"))
      first shouldBe a [UInt]

      info("apply[T <: Data](t: T, init: T): T works")
      val second = Wire(SInt(), SInt(-100))
      second shouldBe a [SInt]

      info("apply[T <: Data](t: T, init: DontCare.type): T works")
      val third = Wire(UInt(), chisel3.DontCare)
      third shouldBe a [UInt]
    }

    ChiselStage.elaborate(new Foo)
  }

  behavior of "Vec"

  it should "support legacy methods" in {
    class Foo extends BasicTester {
      val seq = Seq(Wire(UInt(0, width=4)), Wire(UInt(1, width=4)), Wire(UInt(2, width=4)))
      val vec = Vec(seq)

      info("read works")
      chisel3.assert(vec.read(UInt(0)) === UInt(0))

      info("write works")
      vec.write(UInt(1), UInt(3))
      chisel3.assert(vec.read(UInt(1)) === UInt(3))

      val (_, done) = Counter(Bool(true), 4)
      when (done) { stop }
    }

    assertTesterPasses(new Foo)
  }

  behavior of "Bits methods"

  it should "support legacy methods" in {
    class Foo extends Module {
      val io = new Bundle{}

      val u = UInt(8)
      val s = SInt(-4)

      info("asBits works")
      s.asBits shouldBe a [Bits]

      info("toSInt works")
      u.toSInt shouldBe a [SInt]

      info("toUInt works")
      s.toUInt shouldBe a [UInt]

      info("toBools works")
      s.toBools shouldBe a [Seq[Bool]]
    }

    ChiselStage.elaborate(new Foo)
  }

  it should "properly propagate custom compileOptions in Chisel.Module" in {
    import CompatibilityCustomCompileOptions._
    var result: Foo = null
    ChiselStage.elaborate({result = new Foo; result})
    result.compileOptions should be theSameInstanceAs (customCompileOptions)
  }

  it should "properly set the refs of Records" in {
    class MyRecord extends Record {
      val foo = Vec(1, Bool()).asInput
      val bar = Vec(1, Bool())
      val elements = ListMap("in" -> foo, "out" -> bar)
      def cloneType = (new MyRecord).asInstanceOf[this.type]
    }
    class Foo extends Module {
      val io = IO(new MyRecord)
      io.bar := io.foo
    }
    val verilog = ChiselStage.emitVerilog(new Foo)
    // Check that the names are correct (and that the FIRRTL is valid)
    verilog should include ("assign io_out_0 = io_in_0;")
  }

  it should "ignore .suggestName on field io" in {
    class MyModule extends Module {
      val io = new Bundle {
        val foo = UInt(width = 8).asInput
        val bar = UInt(width = 8).asOutput
      }
      io.suggestName("potato")
      io.bar := io.foo
    }
    val verilog = ChiselStage.emitVerilog(new MyModule)
    verilog should include ("input  [7:0] io_foo")
    verilog should include ("output [7:0] io_bar")
  }

  it should "properly name field io" in {
    class MyModule extends Module {
      val io = new Bundle {
        val foo = UInt(width = 8).asInput
        val bar = UInt(width = 8).asOutput
      }
      val wire = Wire(init = io.foo)
      io.bar := wire
    }
    val verilog = ChiselStage.emitVerilog(new MyModule)
    verilog should include ("input  [7:0] io_foo")
    verilog should include ("output [7:0] io_bar")
  }

}
