package Chisel
import org.scalatest._
import org.scalatest.prop._
import org.scalacheck._

class ChiselSpec extends FlatSpec with PropertyChecks {

  val safeUIntWidth = Gen.choose(1, 31) 
  val safeUInts = Gen.choose(0, (1 << 30))

}

