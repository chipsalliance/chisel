// See LICENSE for license details.

package chiselTests

import org.scalatest._
import org.scalatest.prop._
import org.scalacheck._
import Chisel._
import Chisel.testers._

class ChiselPropSpec extends PropSpec with PropertyChecks {
  def execute(t: => BasicTester): Boolean = TesterDriver.execute(t)
  def elaborate(t: => Module): Circuit = TesterDriver.elaborate(t)

  def popCount(n: Long) = n.toBinaryString.count(_=='1')

  val smallPosInts = Gen.choose(1, 7)
  val safeUIntWidth = Gen.choose(1, 30)
  val safeUInts = Gen.choose(0, (1 << 30))
  val vecSizes = Gen.choose(0, 4)
  val binaryString = for(i <- Arbitrary.arbitrary[Int]) yield "b" + i.toBinaryString
  def enSequence(n: Int) = Gen.containerOfN[List,Boolean](n,Gen.oneOf(true,false))

  def safeUIntN(n: Int) = for {
    w <- smallPosInts
    i <- Gen.containerOfN[List,Int](n, Gen.choose(0, (1 << w) - 1))
  } yield (w, i)
  val safeUInt = for {
    w <- smallPosInts
    i <- Gen.choose(0, (1 << w) - 1)
  } yield (w, i)

  def safeUIntPairN(n: Int) = for {
    w <- smallPosInts
    i <- Gen.containerOfN[List,Int](n, Gen.choose(0, (1 << w) - 1))
    j <- Gen.containerOfN[List,Int](n, Gen.choose(0, (1 << w) - 1))
  } yield (w, i zip j)
  val safeUIntPair= for {
    w <- smallPosInts
    i <- Gen.choose(0, (1 << w) - 1)
    j <- Gen.choose(0, (1 << w) - 1)
  } yield (w, i, j)
}
