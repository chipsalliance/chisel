// SPDX-License-Identifier: Apache-2.0

package firrtl
package benchmark

import firrtl.ir.Circuit
import scala.util.control.NonFatal

package object util {
  def filenameToCircuit(filename: String): Circuit = try {
    proto.FromProto.fromFile(filename)
  } catch {
    case NonFatal(_) => Parser.parseFile(filename, Parser.IgnoreInfo)
  }

  def mean(xs: Iterable[Double]): Double = xs.sum / xs.size

  def median(xs: Iterable[Double]): Double = {
    val size = xs.size
    val sorted = xs.toSeq.sorted
    if (size % 2 == 1) sorted(size / 2)
    else {
      val a = sorted(size / 2)
      val b = sorted((size / 2) - 1)
      (a + b) / 2
    }
  }

  def variance(xs: Iterable[Double]): Double = {
    val avg = mean(xs)
    xs.map(a => math.pow(a - avg, 2)).sum / xs.size
  }

  def stdDev(xs: Iterable[Double]): Double = math.sqrt(variance(xs))
}
