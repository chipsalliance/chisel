// See LICENSE for license details.

package chiselTests

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
      new ArbiterIO(data, 2) shouldBe a [ArbiterIO[_]]
      new LockingRRArbiter(data, 2, 2, None) shouldBe a [LockingRRArbiter[_]]
      new RRArbiter(data, 2) shouldBe a [RRArbiter[_]]
      new Arbiter(data, 2) shouldBe a [Arbiter[_]]
      new Counter(2) shouldBe a [Counter]
      new ValidIO(data) shouldBe a [ValidIO[_]]
      new DecoupledIO(data) shouldBe a [DecoupledIO[_]]
      new QueueIO(data, 2) shouldBe a [QueueIO[_]]
      new Pipe(data, 2) shouldBe a [Pipe[_]]

      FillInterleaved(2, data) shouldBe a [UInt]
      PopCount(data) shouldBe a [UInt]
      Fill(2, data) shouldBe a [UInt]
      Reverse(data) shouldBe a [UInt]
      Cat(data, data) shouldBe a [UInt]
      Log2(data) shouldBe a [UInt]
      unless(Bool(false)) {}
      // 'switch' and 'is' are tested below in Risc
      Counter(2) shouldBe a [Counter]
      DecoupledIO(data) shouldBe a [DecoupledIO[_]]
      val dcd = Decoupled(data)
      dcd shouldBe a [DecoupledIO[_]]
      Queue(dcd) shouldBe a [Queue[_]]
      Enum(data, 2) shouldBe a [List[_]]
      val lfsr16 = LFSR16()
      lfsr16 shouldBe a [UInt]
      lfsr16.getWidth shouldBe (16)
      ListLookup(data, List(data), Array((BitPat("b1"), List(data)))) shouldBe a [List[_]]
      Lookup(data, data, Seq((BitPat("b1"), data))) shouldBe a [List[_]]
      Mux1H(data, Seq(data)) shouldBe a [UInt]
      PriorityMux(Seq(Bool(false)), Seq(data)) shouldBe a [UInt]
      MuxLookup(data, data, Seq((data, data))) shouldBe a [UInt]
      MuxCase(data, Seq((Bool(), data))) shouldBe a [UInt]
      OHToUInt(data) shouldBe a [UInt]
      PriorityEncoder(data) shouldBe a [UInt]
      UIntToOH(data) shouldBe a [UInt]
      PriorityEncoderOH(data) shouldBe a [UInt]
      RegNext(data) shouldBe a [UInt]
      RegInit(data) shouldBe a [UInt]
      RegEnable(data, Bool()) shouldBe a [UInt]
      ShiftRegister(data, 2) shouldBe a [UInt]
      Valid(data) shouldBe a [ValidIO[_]]
      Pipe(Valid(data), 2) shouldBe a [ValidIO[_]]
    }

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
    an [chisel3.core.Binding.ExpectedHardwareException] should be thrownBy {
      elaborate(new Module {
        val io = new Bundle { }
        UInt(INPUT).dir
      })
    }
  }

}
