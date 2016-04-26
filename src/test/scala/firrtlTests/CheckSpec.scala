package firrtlTests

import java.io._
import org.scalatest._
import org.scalatest.prop._
import firrtl.{Parser,Circuit}
import firrtl.passes.{Pass,ToWorkingIR,CheckHighForm,ResolveKinds,InferTypes,CheckTypes,PassExceptions}

class CheckSpec extends FlatSpec with Matchers {
  "Connecting bundles of different types" should "throw an exception" in {
    val passes = Seq(
      ToWorkingIR,
      CheckHighForm)
    val input =
      """circuit Unit :
        |  module Unit :
        |    mem m :
        |      data-type => {a : {b : {flip c : UInt<32>}}}
        |      depth => 32
        |      read-latency => 0
        |      write-latency => 1""".stripMargin
    intercept[PassExceptions] {
      passes.foldLeft(Parser.parse("",input.split("\n").toIterator)) {
        (c: Circuit, p: Pass) => p.run(c)
      }
    }
  }
}
