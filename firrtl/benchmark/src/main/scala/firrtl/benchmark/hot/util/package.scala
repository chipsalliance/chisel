
package firrtl.benchmark.hot

import firrtl.Utils.time
import firrtl.benchmark.util._

package object util {
  def benchmark(nWarmup: Int, nRun: Int)(f: => Unit): Unit = {
    // Warmup
    for (i <- 0 until nWarmup) {
      val (t, res) = time(f)
      println(f"Warmup run $i took $t%.1f ms")
    }

    // Benchmark
    val times: Array[Double] = Array.fill(nRun)(0.0)
    for (i <- 0 until nRun) {
      System.gc
      val (t, res) = time(f)
      times(i) = t
      println(f"Benchmark run $i took $t%.1f ms")
    }

    println(f"Mean:   ${mean(times)}%.1f ms")
    println(f"Median: ${median(times)}%.1f ms")
    println(f"Stddev: ${stdDev(times)}%.1f ms")
  }

}
