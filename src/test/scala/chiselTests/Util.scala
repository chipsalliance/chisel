// See LICENSE for license details.

// Useful utilities for tests

package chiselTests

import chisel3._
import chisel3.experimental.Interval
import chisel3.internal.firrtl.{IntervalRange, KnownBinaryPoint, Width}
import _root_.firrtl.{ir => firrtlir}

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
  val epsilon: Double = 1.0 / math.pow(2.0, binaryPoint.toDouble)

  val (lower, upper) = (intervalRange.lowerBound, intervalRange.upperBound) match {

    case (firrtlir.Closed(lower1), firrtlir.Closed(upper1)) => (lower1, upper1)
    case (firrtlir.Closed(lower1), firrtlir.Open(upper1))   => (lower1, upper1 - epsilon)
    case (firrtlir.Open(lower1),   firrtlir.Closed(upper1)) => (lower1 + epsilon, upper1)
    case (firrtlir.Open(lower1),   firrtlir.Open(upper1))   => (lower1 + epsilon, upper1 - epsilon)
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
    Interval.fromDouble(value.toDouble, width = Width(), binaryPoint = binaryPoint.BP)
  }
}


