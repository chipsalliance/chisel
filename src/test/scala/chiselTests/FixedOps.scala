package chiselTests

import chisel3._
import org.scalatest._
import chisel3.testers.BasicTester

class FixedOps( bitWidth : Int, fracWidth : Int ) extends Module {
  val io = new Bundle {
    val a = Fixed(INPUT, bitWidth, fracWidth)
    val b = Fixed(INPUT, bitWidth, fracWidth)
    val addout = Fixed(OUTPUT, bitWidth, fracWidth)
    val subout = Fixed(OUTPUT, bitWidth, fracWidth)
    val timesout = Fixed(OUTPUT, bitWidth, fracWidth)
    val divout = Fixed(OUTPUT, bitWidth, fracWidth)
    val modout = Fixed(OUTPUT, bitWidth, fracWidth)
    val lshiftout = Fixed(OUTPUT, bitWidth, fracWidth)
    val rshiftout = Fixed(OUTPUT, bitWidth, fracWidth)
    val lessout = Bool(OUTPUT)
    val greatout = Bool(OUTPUT)
    val eqout = Bool(OUTPUT)
    val noteqout = Bool(OUTPUT)
    val lesseqout = Bool(OUTPUT)
    val greateqout = Bool(OUTPUT)
  }

  val a = io.a
  val b = io.b

  io.addout := a + b
  io.subout := a - b
  io.timesout := a *% b
  io.divout := a / Mux(b === Fixed(0, bitWidth, fracWidth), Fixed(1, bitWidth, fracWidth), b)
  io.modout := a % b
  io.lshiftout := (a << b(3, 0))
  io.rshiftout := a >> b.asUInt
  io.lessout := a < b
  io.greatout := a > b
  io.eqout := a === b
  io.noteqout := (a =/= b)
  io.lesseqout := a <= b
  io.greateqout := a >= b
}

class FixedOpsTester( bitWidth : Int, fracWidth : Int, a : BigInt, b : BigInt ) extends BasicTester {
  val dut = Module( new FixedOps( bitWidth, fracWidth ) )
  dut.io.a := Fixed( a, bitWidth, fracWidth )
  dut.io.b := Fixed( b, bitWidth, fracWidth )
  def createFixed ( x  : BigInt ) : Fixed = {
    Fixed( x & ( ( 1 << bitWidth ) - 1 ), bitWidth, fracWidth )
  }
  assert( dut.io.addout === createFixed( a + b ), "Fixed should add correctly" )
  assert( dut.io.subout === createFixed( a - b ), "Fixed should add correctly")
  val timesRes = ((a * b) >> fracWidth) & BigInt((1 << bitWidth) - 1)
  println( "timesRes = " + timesRes )
  assert( dut.io.timesout === createFixed( timesRes ), "Fixed should multiply correctly" )
  // assert( dut.io.divout === createFixed( ((a << fracWidth)/b).toInt ) )
  // assert( dut.io.modout === createFixed( a % b ) )
  assert( dut.io.lshiftout === createFixed( a << (b  & ((1 << 4) - 1) ).toInt ), "Fixed shift left correctly" )
  assert( dut.io.rshiftout === createFixed( a >> b.toInt ) )
  assert( dut.io.lessout === Bool( a < b ) )
  assert( dut.io.greatout === Bool( a > b ) )
  assert( dut.io.eqout === Bool( a == b ) )
  assert( dut.io.noteqout === Bool( a != b ) )
  assert( dut.io.lesseqout === Bool( a <= b ) )
  assert( dut.io.greateqout === Bool( a >= b ) )
  stop()
}

class FixedOpsSpec extends ChiselPropSpec with Matchers {

  val fixedNums = Table(
    ("bw", "fw", "a", "b"),  // First tuple defines column names
    ( 22, 8, 35, 3),  // Subsequent tuples define the data
    ( 8, 4, 17, 17),
    ( 10, 4, 30, 17),
    ( 10, 4, 22, 6),
    ( 10, 5, 40, 16))


  property("FixedOpsTester should return the correct result") {
    forAll (fixedNums) { (bw: Int, fw : Int, a: Int, b: Int) =>
      assertTesterPasses{ new FixedOpsTester(bw, fw, BigInt(a), BigInt(b)) }
    }
  }
}
