
package chiselTests

import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester
import org.scalatest._
import org.scalatest.prop._
import scala.util.Random

class MemShiftRegisterTester( n : Int, useRst : Boolean ) extends BasicTester {
  val in = Wire( UInt( 16.W ) )
  val en = Wire( Bool() )
  val memSr = {
    if ( useRst )
      MemShiftRegister( in, n, 133.U( 16.W ), en )
    else
      MemShiftRegister( in, n, en )
  }
  val srCmp = {
    if ( useRst )
      ShiftRegister( in, n, 133.U( 16.W ), en )
    else
      ShiftRegister( in, n, en )
  }
  val cycs = 3*n
  val cntr = Counter( true.B, cycs )
  val myRand = new Random
  val data = Vec( List.fill( cycs ) { myRand.nextInt( 1 << 15 ).U } )
  val ensRaw = List.fill( cycs ) { (myRand.nextInt(10) == 0) }
  val ens = Vec( ensRaw.map( _.B ) )
  in := data( cntr._1 )
  en := ens( cntr._1 )

  def getEnIdx : Int = {
    var cnt = 0
    for ( eni <- ensRaw.zipWithIndex ) {
      if ( cnt >= n )
        return eni._2
      if ( eni._1 )
        cnt += 1
    }
    cycs
  }

  if ( useRst )
    assert( srCmp === memSr,
      "ShiftRegister and MemShiftRegister should function the same" )
  else {
    when ( cntr._1 >= getEnIdx.U ) {
      assert( srCmp === memSr,
        "ShiftRegister and MemShiftRegister should function the same" )
    }
  }

  when( cntr._2 ) {
    stop()
  }
}

class MemShiftRegisterSpec extends ChiselPropSpec {

  val memParam = Table(
    ("n", "useRst"),  // First tuple defines column names
    ( 3,  false),  // Subsequent tuples define the data
    ( 5,  false),
    ( 25, false),
    ( 73, false),
    ( 3,  true),
    ( 5,  true),
    ( 25, true),
    ( 73, true)
  )

  property("MemShiftRegister should return the correct result") {
    forAll (memParam) { (n : Int, useRst : Boolean ) =>
      assertTesterPasses{ new MemShiftRegisterTester( n, useRst ) }
    }
  }
}
