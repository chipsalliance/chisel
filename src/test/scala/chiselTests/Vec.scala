// SPDX-License-Identifier: Apache-2.0

package chiselTests

import org.scalacheck._

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.testers.BasicTester
import chisel3.util._
import org.scalacheck.Shrink
import scala.annotation.tailrec

class LitTesterMod(vecSize: Int) extends Module {
  val io = IO(new Bundle {
    val out = Output(Vec(vecSize, UInt()))
  })
  io.out := VecInit(Seq.fill(vecSize){0.U})
}

class RegTesterMod(vecSize: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(vecSize, UInt()))
    val out = Output(Vec(vecSize, UInt()))
  })
  val vecReg = RegNext(io.in, VecInit(Seq.fill(vecSize){0.U}))
  io.out := vecReg
}

class IOTesterMod(vecSize: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(vecSize, UInt()))
    val out = Output(Vec(vecSize, UInt()))
  })
  io.out := io.in
}

class OneBitUnitRegVec extends Module {
  val io = IO(new Bundle {
    val out = Output(UInt(1.W))
  })
  val oneBitUnitRegVec = Reg(Vec(1, UInt(1.W)))
  oneBitUnitRegVec(0) := 1.U(1.W)
  io.out := oneBitUnitRegVec(0)
}

class LitTester(w: Int, values: List[Int]) extends BasicTester {
  val dut = Module(new LitTesterMod(values.length))
  for (a <- dut.io.out)
    assert(a === 0.U)
  stop()
}

class RegTester(w: Int, values: List[Int]) extends BasicTester {
  val v = VecInit(values.map(_.U(w.W)))
  val dut = Module(new RegTesterMod(values.length))
  val doneReg = RegInit(false.B)
  dut.io.in := v
  when (doneReg) {
    for ((a,b) <- dut.io.out.zip(values))
      assert(a === b.U)
    stop()
  } .otherwise {
    doneReg := true.B
    for (a <- dut.io.out)
      assert(a === 0.U)
  }
}

class IOTester(w: Int, values: List[Int]) extends BasicTester {
  val v = VecInit(values.map(_.U(w.W))) // Does this need a Wire? No. It's a Vec of Lits and hence synthesizeable.
  val dut = Module(new IOTesterMod(values.length))
  dut.io.in := v
  for ((a,b) <- dut.io.out.zip(values)) {
    assert(a === b.U)
  }
  stop()
}

class IOTesterModFill(vecSize: Int) extends Module {
  // This should generate a BindingException when we attempt to wire up the Vec.fill elements
  //  since they're pure types and hence unsynthesizeable.
  val io = IO(new Bundle {
    val in = Input(VecInit(Seq.fill(vecSize) {UInt()}))
    val out = Output(VecInit(Seq.fill(vecSize) {UInt()}))
  })
  io.out := io.in
}

class ValueTester(w: Int, values: List[Int]) extends BasicTester {
  val v = VecInit(values.map(_.asUInt(w.W)))
  for ((a,b) <- v.zip(values)) {
    assert(a === b.asUInt)
  }
  stop()
}

class TabulateTester(n: Int) extends BasicTester {
  val v = VecInit(Range(0, n).map(i => (i*2).asUInt))
  val x = VecInit(Array.tabulate(n){ i => (i*2).asUInt })
  val u = VecInit.tabulate(n)(i => (i*2).asUInt)

  assert(v.asUInt() === x.asUInt())
  assert(v.asUInt() === u.asUInt())
  assert(x.asUInt() === u.asUInt())

  stop()
}

class FillTester(n: Int, value: Int) extends BasicTester {
  val x = VecInit(Array.fill(n)(value.U))
  val u = VecInit.fill(n)(value.U)

  assert(x.asUInt() === u.asUInt(), s"Expected Vec to be filled like $x, instead VecInit.fill created $u")
  stop()
}

object VecMultiDimTester { 
  
  @tailrec
  private def assert2DIsCorrect(n: Int, arr: Vec[Vec[UInt]], compArr: Seq[Seq[Int]]): Unit = {
    val compareRow = arr(n) zip compArr(n)
    compareRow.foreach (x => assert(x._1 === x._2.U))
    if (n != 0) assert2DIsCorrect(n-1, arr, compArr)
  }

  @tailrec
  private def assert3DIsCorrect(n: Int, m: Int, arr: Vec[Vec[Vec[UInt]]], compArr: Seq[Seq[Seq[Int]]]): Unit = {
    assert2DIsCorrect(m-1, arr(n), compArr(n))
    if (n != 0) assert3DIsCorrect(n-1, m, arr, compArr)
  }

  class TabulateTester2D(n: Int, m: Int) extends BasicTester {
    def gen(x: Int, y: Int): UInt = (x+y).asUInt
    def genCompVec(x: Int, y:Int): Int = x+y
    val vec = VecInit.tabulate(n, m){ gen }
    val compArr = Seq.tabulate(n,m){ genCompVec }
    
    assert2DIsCorrect(n-1, vec, compArr)
    stop()
  }

  class TabulateTester3D(n: Int, m: Int, p: Int) extends BasicTester {
    def gen(x: Int, y: Int, z: Int): UInt = (x+y+z).asUInt
    def genCompVec(x: Int, y:Int, z: Int): Int = x+y+z
    val vec = VecInit.tabulate(n, m, p){ gen }
    val compArr = Seq.tabulate(n, m, p){ genCompVec }

    assert3DIsCorrect(n-1, m, vec, compArr)
    stop()
  }

  class Fill2DTester(n: Int, m: Int, value: Int) extends BasicTester {
    val u = VecInit.fill(n,m)(value.U)
    val compareArr = Seq.fill(n,m)(value)
    
    assert2DIsCorrect(n-1, u, compareArr)
    stop()
  }

  class Fill3DTester(n: Int, m: Int, p: Int, value: Int) extends BasicTester {
    val u = VecInit.fill(n,m,p)(value.U)
    val compareArr = Seq.fill(n,m,p)(value)

    assert3DIsCorrect(n-1, m, u, compareArr)
    stop()
  }

  class BidirectionalTester2DFill(n: Int, m: Int) extends BasicTester {
    val mod = Module(new PassthroughModule)
    val vec2D = VecInit.fill(n, m)(mod.io)
    for {
      vec1D <- vec2D
      module <- vec1D
    } yield {
      module <> Module(new PassthroughModuleTester).io
    }
    stop()
  }

  class BidirectionalTester3DFill(n: Int, m: Int, p: Int) extends BasicTester {
    val mod = Module(new PassthroughModule)
    val vec3D = VecInit.fill(n, m, p)(mod.io)
    
    for {
      vec2D <- vec3D
      vec1D <- vec2D
      module <- vec1D
    } yield {
      module <> (Module(new PassthroughModuleTester).io)
    }
    stop()
  }
  
  class TabulateModuleTester(value: Int) extends Module {
    val io = IO(Flipped(new PassthroughModuleIO))
    // This drives the input of a PassthroughModule
    io.in := value.U
  }

  class BidirectionalTester2DTabulate(n: Int, m: Int) extends BasicTester {
    val vec2D = VecInit.tabulate(n, m) { (x, y) =>  Module(new TabulateModuleTester(x + y + 1)).io}

    for {
      x <- 0 until n
      y <- 0 until m 
    } yield {
      val value = x + y + 1
      val receiveMod = Module(new PassthroughModule).io
      vec2D(x)(y) <> receiveMod
      assert(receiveMod.out === value.U)
    }
    stop()
  }

  class BidirectionalTester3DTabulate(n: Int, m: Int, p: Int) extends BasicTester {
    val vec3D = VecInit.tabulate(n, m, p) { (x, y, z) => Module(new TabulateModuleTester(x + y + z + 1)).io }

    for {
      x <- 0 until n
      y <- 0 until m
      z <- 0 until p
    } yield {
      val value = x + y + z + 1
      val receiveMod = Module(new PassthroughModule).io
      vec3D(x)(y)(z) <> receiveMod
      assert(receiveMod.out === value.U)
    }
    stop()
  }
}

class IterateTester(start: Int, len: Int)(f: UInt => UInt) extends BasicTester {
  val controlVec = VecInit(Seq.iterate(start.U, len)(f))
  val testVec = VecInit.iterate(start.U, len)(f)
  assert(controlVec.asUInt() === testVec.asUInt(), s"Expected Vec to be filled like $controlVec, instead creaeted $testVec\n")
  stop()
}

class ShiftRegisterTester(n: Int) extends BasicTester {
  val (cnt, wrap) = Counter(true.B, n*2)
  val shifter = Reg(Vec(n, UInt((log2Ceil(n) max 1).W)))
  (shifter, shifter drop 1).zipped.foreach(_ := _)
  shifter(n-1) := cnt
  when (cnt >= n.asUInt) {
    val expected = cnt - n.asUInt
    assert(shifter(0) === expected)
  }
  when (wrap) {
    stop()
  }
}

class HugeVecTester(n: Int) extends BasicTester {
  require(n > 0)
  val myVec = Wire(Vec(n, UInt()))
  myVec.foreach { x =>
    x := 123.U
    assert(x === 123.U)
  }
  stop()
}

class OneBitUnitRegVecTester extends BasicTester {
  val dut = Module(new OneBitUnitRegVec)
  assert(dut.io.out === 1.U)
  stop()
}

class ZeroEntryVecTester extends BasicTester {
  require(Vec(0, Bool()).getWidth == 0)

  val bundleWithZeroEntryVec = new Bundle {
    val foo = Bool()
    val bar = Vec(0, Bool())
  }
  require(0.U.asTypeOf(bundleWithZeroEntryVec).getWidth == 1)
  require(bundleWithZeroEntryVec.asUInt.getWidth == 1)

  val m = Module(new Module {
    val io = IO(Output(bundleWithZeroEntryVec))
    io.foo := false.B
  })
  WireDefault(m.io.bar)

  stop()
}

class PassthroughModuleTester extends Module {
  val io = IO(Flipped(new PassthroughModuleIO))
  // This drives the input of a PassthroughModule
  io.in := 123.U
  assert(io.out === 123.U)
}

class ModuleIODynamicIndexTester(n: Int) extends BasicTester {
  val duts = VecInit.fill(n)(Module(new PassthroughModule).io)
  val tester = Module(new PassthroughModuleTester)

  val (cycle, done) = Counter(true.B, n)
  for ((m, i) <- duts.zipWithIndex) {
    when (cycle =/= i.U) {
      m.in := 0.U  // default
      assert(m.out === 0.U)
    } .otherwise {
      m.in := DontCare
    }
  }
  // only connect one dut per cycle
  duts(cycle) <> tester.io
  assert(duts(cycle).out === 123.U)

  when (done) { stop() }
}

class ReduceTreeTester() extends BasicTester {
  class FooIO[T <: Data](n: Int, private val gen: T) extends Bundle {
    val in = Flipped(Vec(n, new DecoupledIO(gen)))
    val out = new DecoupledIO(gen)
  }

  class Foo[T <: Data](n: Int, private val gen: T) extends Module {
    val io = IO(new FooIO(n, gen))

    def foo(a: DecoupledIO[T], b: DecoupledIO[T]) = {
      a.ready := true.B
      b.ready := true.B
      val out = Wire(new DecoupledIO(gen))

      out.valid := true.B

      val regSel = RegInit(false.B)
      out.bits := Mux(regSel, a.bits, b.bits)
      out.ready := a.ready
      out
    }

    io.out <> io.in.reduceTree(foo)
  }

  val dut = Module(new Foo(5, UInt(5.W)))
  dut.io := DontCare
  stop()
}

class VecSpec extends ChiselPropSpec with Utils {
  // Disable shrinking on error.
  implicit val noShrinkListVal = Shrink[List[Int]](_ => Stream.empty)
  implicit val noShrinkInt = Shrink[Int](_ => Stream.empty)

  property("Vecs should be assignable") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new ValueTester(w, v) }
    }
  }

  property("Vecs should be passed through vec IO") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new IOTester(w, v) }
    }
  }

  property("Vec.fill with a pure type should generate an exception") {
    // We don't really need a sequence of random widths here, since any should throw an exception.
    forAll(safeUIntWidth) { case(w: Int) =>
      an[BindingException] should be thrownBy extractCause[BindingException] {
        ChiselStage.elaborate(new IOTesterModFill(w))
      }
    }
  }

  property("A Reg of a Vec should operate correctly") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new RegTester(w, v) }
    }
  }

  property("A Vec of lit should operate correctly") {
    forAll(safeUIntN(8)) { case(w: Int, v: List[Int]) =>
      assertTesterPasses{ new LitTester(w, v) }
    }
  }

  property("VecInit should tabulate correctly") {
    forAll(smallPosInts) { (n: Int) => assertTesterPasses{ new TabulateTester(n) } }
  }

  property("VecInit should tabulate 2D vec correctly") {
    forAll(smallPosInts, smallPosInts) { (n: Int, m: Int) => assertTesterPasses { new VecMultiDimTester.TabulateTester2D(n, m) } }
  }

  property("VecInit should tabulate 3D vec correctly") {
      forAll(smallPosInts, smallPosInts, smallPosInts) { (n: Int, m: Int, p: Int) => assertTesterPasses{ new VecMultiDimTester.TabulateTester3D(n, m, p) } }
  }

  property("VecInit should fill correctly") {
    forAll(smallPosInts, Gen.choose(0, 50)) { (n: Int, value: Int) => assertTesterPasses{ new FillTester(n, value) } }
  }

  property("VecInit should fill 2D vec correctly") {
    forAll(smallPosInts, smallPosInts, Gen.choose(0, 50)) { (n: Int, m: Int, value: Int) => assertTesterPasses{ new VecMultiDimTester.Fill2DTester(n, m, value) } }
  }
  
  property("VecInit should fill 3D vec correctly") {
    forAll(smallPosInts, smallPosInts, smallPosInts, Gen.choose(0, 50)) { (n: Int, m: Int, p: Int, value: Int) => assertTesterPasses{ new VecMultiDimTester.Fill3DTester(n, m, p, value) } }
  }

  property("VecInit should support 2D fill bidirectional wire connection") {
    forAll(smallPosInts, smallPosInts) { (n: Int, m: Int) => assertTesterPasses{ new VecMultiDimTester.BidirectionalTester2DFill(n, m) }} 
  }
  
  property("VecInit should support 3D fill bidirectional wire connection") {
    forAll(smallPosInts, smallPosInts, smallPosInts) { (n: Int, m: Int, p: Int) => assertTesterPasses{ new VecMultiDimTester.BidirectionalTester3DFill(n, m, p) }}
  }

  property("VecInit should support 2D tabulate bidirectional wire connection") {
    forAll(smallPosInts, smallPosInts) { (n: Int, m: Int) => assertTesterPasses{ new VecMultiDimTester.BidirectionalTester2DTabulate(n, m) }} 
  }
  
  property("VecInit should support 3D tabulate bidirectional wire connection") {
    forAll(smallPosInts, smallPosInts, smallPosInts) { (n: Int, m: Int, p: Int) => assertTesterPasses{ new VecMultiDimTester.BidirectionalTester3DTabulate(n, m, p) }}
  }
  
  property("VecInit should iterate correctly") {
    forAll(Gen.choose(1, 10), smallPosInts) { (start: Int, len: Int) => assertTesterPasses{ new IterateTester(start, len)(x => x + 50.U)}}
  }

  property("Regs of vecs should be usable as shift registers") {
    forAll(smallPosInts) { (n: Int) => assertTesterPasses{ new ShiftRegisterTester(n) } }
  }

  property("Infering widths on huge Vecs should not cause a stack overflow") {
    assertTesterPasses { new HugeVecTester(10000) }
  }

  property("A Reg of a Vec of a single 1 bit element should compile and work") {
    assertTesterPasses{ new OneBitUnitRegVecTester }
  }

  property("A Vec with zero entries should compile and have zero width") {
    assertTesterPasses{ new ZeroEntryVecTester }
  }

  property("Dynamic indexing of a Vec of Module IOs should work") {
    assertTesterPasses{ new ModuleIODynamicIndexTester(4) }
  }

  property("It should be possible to bulk connect a Vec and a Seq") {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val out = Output(Vec(4, UInt(8.W)))
      })
      val seq = Seq.fill(4)(0.U)
      io.out <> seq
    })
  }

  property("Bulk connecting a Vec and Seq of different sizes should report a ChiselException") {
    a [ChiselException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate(new Module {
        val io = IO(new Bundle {
          val out = Output(Vec(4, UInt(8.W)))
        })
        val seq = Seq.fill(5)(0.U)
        io.out <> seq
      })
    }
  }

  property("It should be possible to initialize a Vec with DontCare") {
    ChiselStage.elaborate(new Module {
      val io = IO(new Bundle {
        val out = Output(Vec(4, UInt(8.W)))
      })
      io.out := VecInit(Seq(4.U, 5.U, DontCare, 2.U))
    })
  }

  property("Indexing a Chisel type Vec by a hardware type should give a sane error message") {
    a [ExpectedHardwareException] should be thrownBy extractCause[ChiselException] {
      ChiselStage.elaborate{
        new Module {
          val io = IO(new Bundle{})
          val foo = Vec(2, Bool())
          foo(0.U) := false.B
        }}
    }
  }

  property("reduceTree should preserve input/output type") {
      assertTesterPasses { new ReduceTreeTester() }
  }
}
