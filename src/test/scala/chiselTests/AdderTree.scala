package chiselTests

import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester
import org.scalatest._
import org.scalacheck._
import org.scalatest.prop._
import scala.collection.mutable.ArrayBuffer

class AdderTree[ T <: Bits with Num[T] ]( genType : T, vecSize : Int ) extends Module {
  val io = new Bundle {
    val numIn = Vec( vecSize, genType ).asInput
    val numOut = genType.cloneType.asOutput
  }
  io.numOut := io.numIn.reduceTree(
    (a : T, b : T) => ( a + b ) )
}

class AdderTreeTester( bitWidth : Int, numsToAdd : List[Int] ) extends BasicTester {
  println("numsToAdd.size = " + numsToAdd.size )
  val genType = UInt( width = bitWidth )
  val dut = Module( new AdderTree( genType, numsToAdd.size ) )
  dut.io.numIn := Vec(numsToAdd.map(x => UInt(x, bitWidth)))
  val sumCorrect = dut.io.numOut === UInt( numsToAdd.reduce(_+_) % (1 << bitWidth ), bitWidth )
  assert(sumCorrect)
  stop()
}

class AdderTreeSpec extends ChiselPropSpec {
  property("All numbers should be added correctly by an Adder Tree") {
    forAll( safeUIntN( 20 ) ) {
      case(w: Int, v: List[Int]) => {
        whenever ( v.size > 0 && w > 0 ) {
          assertTesterPasses{ new AdderTreeTester( w, v.map(x => math.abs(x) % ( 1 << w ) ).toList ) }
        }
      }
    }
  }
}
