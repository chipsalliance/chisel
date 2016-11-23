// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util._

class DeduplicationSpec extends PerformancePathologySpec {
  /** Generates lots of identical Queues */
  class DedupQueue(n: Int) extends Module {
    require(n >= 0)
    val io = IO(new Bundle {
      val in = Flipped(Decoupled(UInt(64.W)))
      val out = Decoupled(UInt(64.W))
    })

    var ref = io.in
    for (i <- 0 until n) {
      ref = Queue(ref, 8)
    }
    io.out <> ref
  }

  def timeout: Int = 10
  def warmupConfig = () => new DedupQueue(8)
  def benchmarkConfig = () => new DedupQueue(128)

  // Check Verilog result
  it should "actually do deduplication" in {
    val ModuleRegex = """\s*module\s+(\w+)\b.*""".r

    val modules = verilogResult split "\n" collect { case ModuleRegex(name) => name }
    assert(modules.size === 2, "Deduplication must actually happen!")
  }
}
