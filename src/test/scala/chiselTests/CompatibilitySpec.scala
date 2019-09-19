// See LICENSE for license details.

package chiselTests

import chisel3.testers.BasicTester

import org.scalacheck.Gen
import org.scalatest.prop.GeneratorDrivenPropertyChecks

class CompatibiltySpec extends ChiselFlatSpec with GeneratorDrivenPropertyChecks {
  import Chisel._

  behavior of "Chisel compatibility layer"

  it should "accept direction arguments" in {
    elaborate(new Module {
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
    log2Down(value) shouldBe ((1 max BigInt(value - 1).bitLength) - (if (value > 0 && ((value & (value - 1)) == 0)) 0 else 1)) // scalastyle:ignore line.size.limit
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
      unless(Bool(false)) {}
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
    elaborate { new Dummy }
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
    elaborate { new Chisel2CompatibleRisc }
  }

  it should "not try to assign directions to Analog" in {
    elaborate(new Module {
      val io = new Bundle {
        val port = chisel3.experimental.Analog(32.W)
      }
    })
  }


  class SmallBundle extends Bundle {
    val f1 = UInt(width = 4)
    val f2 = UInt(width = 5)
    override def cloneType: this.type = (new SmallBundle).asInstanceOf[this.type]
  }
  class BigBundle extends SmallBundle {
    val f3 = UInt(width = 6)
    override def cloneType: this.type = (new BigBundle).asInstanceOf[this.type]
  }

  // scalastyle:off line.size.limit
  "A Module with missing bundle fields when compiled with the Chisel compatibility package" should "not throw an exception" in {

    class ConnectFieldMismatchModule extends Module {
      val io = new Bundle {
        val in = (new SmallBundle).asInput
        val out = (new BigBundle).asOutput
      }
      io.out := io.in
    }
    elaborate { new ConnectFieldMismatchModule() }
  }

  "A Module in which a Reg is created with a bound type when compiled with the Chisel compatibility package" should "not throw an exception" in {

    class CreateRegFromBoundTypeModule extends Module {
      val io = new Bundle {
        val in = (new SmallBundle).asInput
        val out = (new BigBundle).asOutput
      }
      val badReg = Reg(UInt(7, width=4))
    }
    elaborate { new CreateRegFromBoundTypeModule() }
  }

  "A Module with unwrapped IO when compiled with the Chisel compatibility package" should "not throw an exception" in {

    class RequireIOWrapModule extends Module {
      val io = new Bundle {
        val in = UInt(width = 32).asInput
        val out = Bool().asOutput
      }
      io.out := io.in(1)
    }
    elaborate { new RequireIOWrapModule() }
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
    elaborate { new SwappedConnectionModule() }
  }

  "A Module with directionless connections when compiled with the Chisel compatibility package" should "not throw an exception" in {

    class SimpleModule extends Module {
      val io = new Bundle {
        val in = (UInt(width = 3)).asInput
        val out = (UInt(width = 4)).asOutput
      }
      val noDir = Wire(UInt(width = 3))
    }

    class DirectionLessConnectionModule extends SimpleModule {
      val a = UInt(0, width = 3)
      val b = Wire(UInt(width = 3))
      val child = Module(new SimpleModule)
      b := child.noDir
    }
    elaborate { new DirectionLessConnectionModule() }
  }

  "Vec ports" should "give default directions to children so they can be used in chisel3.util" in {
    import Chisel._
    elaborate(new Module {
      val io = new Bundle {
        val in = Vec(1, UInt(width = 8)).flip
        val out = UInt(width = 8)
      }
      io.out := RegEnable(io.in(0), true.B)
    })
  }

  "Reset" should "still walk, talk, and quack like a Bool" in {
    import Chisel._
    elaborate(new Module {
      val io = new Bundle {
        val in = Bool(INPUT)
        val out = Bool(OUTPUT)
      }
      io.out := io.in && reset
    })
  }

  "Data.dir" should "give the correct direction for io" in {
    import Chisel._
    elaborate(new Module {
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
    an [chisel3.ExpectedHardwareException] should be thrownBy {
      elaborate(new Module {
        val io = new Bundle { }
        UInt(INPUT).dir
      })
    }
  }

  "Mux return value" should "be able to be used on the RHS" in {
    import Chisel._
    elaborate(new Module {
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
    elaborate(new Module {
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
  // scalastyle:on line.size.limit

  behavior of "BitPat"

  it should "support old operators" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      info("Deprecated method DC hasn't been removed")
      val bp = BitPat.DC(4)

      info("BitPat != UInt is a Bool")
      (bp != UInt(4)) shouldBe a [Bool]

      /* This test does not work, but I'm not sure it's supposed to? It does *not* work on chisel3. */
      // info("UInt != BitPat is a Bool")
      // (UInt(4) != bp) shouldBe a [Bool]
    }

    elaborate(new Foo)
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

    elaborate(new Foo)
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

    elaborate(new Foo)
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

    elaborate(new Foo)
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

    elaborate(new Foo)
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

    elaborate(new Foo)
  }

  behavior of "debug"

  it should "still exist" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      val data = UInt(width = 2)
      debug(data)
    }

    elaborate(new Foo)
  }

  behavior of "Data methods"

  it should "support legacy methods" in {
    class Foo extends Module {
      val io = IO(new Bundle{})

      info("litArg works")
      UInt(width=3).litArg() should be (None)
      UInt(0, width=3).litArg() should be (Some(chisel3.internal.firrtl.ULit(0, 3.W)))

      info("toBits works")
      val wire = Wire(UInt(width=4))
      Vec.fill(4)(wire).toBits.getWidth should be (wire.getWidth * 4)
    }

    elaborate(new Foo)
  }

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

    elaborate(new Foo)
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

      info("toBools works")
      u.toBools shouldBe a [Seq[Bool]]

      info("asBits works")
      s.asBits shouldBe a [Bits]

      info("toSInt works")
      u.toSInt shouldBe a [SInt]

      info("toUInt works")
      s.toUInt shouldBe a [UInt]

      info("toBool works")
      UInt(1).toBool shouldBe a [Bool]
    }

    elaborate(new Foo)
  }

  behavior of "UInt"

  it should "support legacy methods" in {
    class Foo extends Module {
      val io = new Bundle{}

      info("!= works")
      (UInt(1) != UInt(1)) shouldBe a [Bool]
    }

    elaborate(new Foo)
  }

  behavior of "SInt"

  it should "support legacy methods" in {
    class Foo extends Module {
      val io = new Bundle{}

      info("!= works")
      (SInt(-1) != SInt(-1)) shouldBe a [Bool]
    }

    elaborate(new Foo)
  }

}
