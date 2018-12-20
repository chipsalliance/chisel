// Useful utilities for tests

package chiselTests

import chisel3._
import chisel3.experimental._
import chisel3.internal.firrtl.{IntervalRange, KnownBinaryPoint}
import _root_.firrtl.{ir => firrtlir}
import org.scalatest.{FreeSpec, Matchers}

class PassthroughModuleIO extends Bundle {
  val in = Input(UInt(32.W))
  val out = Output(UInt(32.W))
}

trait AbstractPassthroughModule extends RawModule {
  val io = IO(new PassthroughModuleIO)
  io.out := io.in
}

class PassthroughModule extends Module with AbstractPassthroughModule
class PassthroughMultiIOModule extends MultiIOModule with AbstractPassthroughModule
class PassthroughRawModule extends RawModule with AbstractPassthroughModule

case class ScalaIntervalSimulator(intervalRange: IntervalRange) {
  val binaryPoint: Int = intervalRange.binaryPoint.asInstanceOf[KnownBinaryPoint].value
  val epsilon = 1.0 / math.pow(2.0, binaryPoint.toDouble)

  val (lower, upper) = (intervalRange.lowerBound, intervalRange.upperBound) match {

    case (firrtlir.Closed(lower), firrtlir.Closed(upper)) => (lower, upper)
    case (firrtlir.Closed(lower), firrtlir.Open(upper))   => (lower, upper - epsilon)
    case (firrtlir.Open(lower),   firrtlir.Closed(upper)) => (lower + epsilon, upper)
    case (firrtlir.Open(lower),   firrtlir.Open(upper))   => (lower + epsilon, upper - epsilon)
    case _ =>
      throw new Exception(s"lower and upper bounds must be defined, range here is $intervalRange")
  }

  def clip(value: BigDecimal): BigDecimal = {

    if (value < lower) {
      lower
    }
    else if (value > upper) {
      upper
    }
    else {
      value
    }
  }

  def wrap(value: BigDecimal): BigDecimal = {

    if (value < lower) {
      upper + (value - lower) + epsilon
    }
    else if (value > upper) {
      ((value - upper) - epsilon) + lower
    }
    else {
      value
    }
  }

  def allValues: Iterator[BigDecimal] = {
    (lower to upper by epsilon).toIterator
  }

  def makeLit(value: BigDecimal): Interval = {
    Interval.fromDouble(value.toDouble, binaryPoint = binaryPoint)
  }
}

class ScalaIntervalSimulatorSpec extends FreeSpec with Matchers {
  "clip tests" - {
    "Should work for closed ranges" in {
      val sim = ScalaIntervalSimulator(range"[2,4]")
      sim.clip(BigDecimal(1.0)) should be (2.0)
      sim.clip(BigDecimal(2.0)) should be (2.0)
      sim.clip(BigDecimal(3.0)) should be (3.0)
      sim.clip(BigDecimal(4.0)) should be (4.0)
      sim.clip(BigDecimal(5.0)) should be (4.0)
    }
    "Should work for closed ranges with binary point" in {
      val sim = ScalaIntervalSimulator(range"[2,6].2")
      sim.clip(BigDecimal(1.75)) should be (2.0)
      sim.clip(BigDecimal(2.0))  should be (2.0)
      sim.clip(BigDecimal(2.25)) should be (2.25)
      sim.clip(BigDecimal(2.5))  should be (2.5)
      sim.clip(BigDecimal(5.75)) should be (5.75)
      sim.clip(BigDecimal(6.0))  should be (6.0)
      sim.clip(BigDecimal(6.25)) should be (6.0)
      sim.clip(BigDecimal(6.5))  should be (6.0)
      sim.clip(BigDecimal(8.5))  should be (6.0)
    }
    "Should work for open ranges" in {
      val sim = ScalaIntervalSimulator(range"(2,4)")
      sim.clip(BigDecimal(1.0)) should be (3.0)
      sim.clip(BigDecimal(2.0)) should be (3.0)
      sim.clip(BigDecimal(3.0)) should be (3.0)
      sim.clip(BigDecimal(4.0)) should be (3.0)
      sim.clip(BigDecimal(5.0)) should be (3.0)
    }
    "Should work for open ranges with binary point" in {
      val sim = ScalaIntervalSimulator(range"(2,6).2")
      sim.clip(BigDecimal(1.75)) should be (2.25)
      sim.clip(BigDecimal(2.0))  should be (2.25)
      sim.clip(BigDecimal(2.25)) should be (2.25)
      sim.clip(BigDecimal(2.5))  should be (2.5)
      sim.clip(BigDecimal(5.75)) should be (5.75)
      sim.clip(BigDecimal(6.0))  should be (5.75)
      sim.clip(BigDecimal(6.25)) should be (5.75)
      sim.clip(BigDecimal(6.5))  should be (5.75)
      sim.clip(BigDecimal(8.5))  should be (5.75)
    }
  }
  "wrap tests" - {
    "Should work for closed ranges" in {
      val sim = ScalaIntervalSimulator(range"[2,6]")
      sim.wrap(BigDecimal(1.0)) should be (6.0)
      sim.wrap(BigDecimal(2.0)) should be (2.0)
      sim.wrap(BigDecimal(3.0)) should be (3.0)
      sim.wrap(BigDecimal(4.0)) should be (4.0)
      sim.wrap(BigDecimal(5.0)) should be (5.0)
      sim.wrap(BigDecimal(6.0)) should be (6.0)
      sim.wrap(BigDecimal(7.0)) should be (2.0)
    }
    "Should work for closed ranges with binary point" in {
      val sim = ScalaIntervalSimulator(range"[2,6].2")
      sim.wrap(BigDecimal(1.75)) should be (6.0)
      sim.wrap(BigDecimal(2.0))  should be (2.0)
      sim.wrap(BigDecimal(2.25)) should be (2.25)
      sim.wrap(BigDecimal(2.5))  should be (2.5)
      sim.wrap(BigDecimal(5.75)) should be (5.75)
      sim.wrap(BigDecimal(6.0))  should be (6.0)
      sim.wrap(BigDecimal(6.25))  should be (2.0)
      sim.wrap(BigDecimal(6.5))  should be (2.25)
    }
    "Should work for open ranges" in {
      val sim = ScalaIntervalSimulator(range"(2,6)")
      sim.wrap(BigDecimal(1.0)) should be (4.0)
      sim.wrap(BigDecimal(2.0)) should be (5.0)
      sim.wrap(BigDecimal(3.0)) should be (3.0)
      sim.wrap(BigDecimal(4.0)) should be (4.0)
      sim.wrap(BigDecimal(5.0)) should be (5.0)
      sim.wrap(BigDecimal(6.0)) should be (3.0)
      sim.wrap(BigDecimal(7.0)) should be (4.0)
    }
    "Should work for open ranges with binary point" in {
      val sim = ScalaIntervalSimulator(range"(2,6).2")
      sim.wrap(BigDecimal(1.75)) should be (5.5)
      sim.wrap(BigDecimal(2.0)) should be (5.75)
      sim.wrap(BigDecimal(2.25)) should be (2.25)
      sim.wrap(BigDecimal(2.5)) should be (2.5)
      sim.wrap(BigDecimal(5.75)) should be (5.75)
      sim.wrap(BigDecimal(6.0)) should be (2.25)
      sim.wrap(BigDecimal(6.25)) should be (2.5)
      sim.wrap(BigDecimal(7.0)) should be (3.25)
    }
  }
}


