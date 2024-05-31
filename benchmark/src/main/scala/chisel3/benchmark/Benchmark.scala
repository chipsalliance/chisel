// SPDX-License-Identifier: Apache-2.0

package chisel3.benchmark

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations._
import org.openjdk.jmh.infra.Blackhole

import chisel3._
import chisel3.util.BitPat

// Run with:
// ./mill benchmark.runJmh
class ChiselBenchmark {

  // This is just an example, copy-paste and modify as appropriate
  // Typically 10 iterations for both warmup and measurement is better
  @Benchmark
  @OutputTimeUnit(TimeUnit.MICROSECONDS)
  @Warmup(iterations = 3)
  @Measurement(iterations = 3)
  @Fork(value = 1)
  @Threads(value = 1)
  def BitPatFromUInt(blackHole: Blackhole): BitPat = {
    val x = BitPat(0xdeadbeefL.U)
    // Blackhole consuming the value prevents certain JVM optimizations
    blackHole.consume(x)
    // Returning the value is usually a good idea as well
    x
  }
}
