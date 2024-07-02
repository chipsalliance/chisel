// SPDX-License-Identifier: Apache-2.0

package chisel3.benchmark

import java.util.concurrent.TimeUnit
import org.openjdk.jmh.annotations._
import org.openjdk.jmh.infra.Blackhole

import circt.stage.ChiselStage
import chisel3._
import chisel3.util.BitPat

object ChiselBenchmark {
  class MyBundle extends Bundle {
    val a, b, c, d, e, f, g = UInt(8.W)
  }
  class Foo(n: Int) extends Module {
    val in = IO(Input(Vec(n, new MyBundle)))
    val out = IO(Output(Vec(n, new MyBundle)))

    out :#= in
  }
}

// Run with:
// ./mill benchmark.runJmh
class ChiselBenchmark {

  // // This is just an example, copy-paste and modify as appropriate
  // // Typically 10 iterations for both warmup and measurement is better
  // @Benchmark
  // @OutputTimeUnit(TimeUnit.MICROSECONDS)
  // @Warmup(iterations = 3)
  // @Measurement(iterations = 3)
  // @Fork(value = 1)
  // @Threads(value = 1)
  // def BitPatFromUInt(blackHole: Blackhole): BitPat = {
  //   val x = BitPat(0xdeadbeefL.U)
  //   // Blackhole consuming the value prevents certain JVM optimizations
  //   blackHole.consume(x)
  //   // Returning the value is usually a good idea as well
  //   x
  // }

  // This is just an example, copy-paste and modify as appropriate
  // Typically 10 iterations for both warmup and measurement is better
  @Benchmark
  @OutputTimeUnit(TimeUnit.MILLISECONDS)
  @Warmup(iterations = 3)
  @Measurement(iterations = 3)
  @Fork(value = 1)
  @Threads(value = 1)
  def VecOfBundles(blackHole: Blackhole): String = {
    val x = ChiselStage.emitCHIRRTL(new ChiselBenchmark.Foo(10))
    // Blackhole consuming the value prevents certain JVM optimizations
    blackHole.consume(x)
    // Returning the value is usually a good idea as well
    x
  }
}
