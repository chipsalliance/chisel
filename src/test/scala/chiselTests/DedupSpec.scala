// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util._

class DedupIO extends Bundle {
  val in = Flipped(Decoupled(UInt(32.W)))
  val out = Decoupled(UInt(32.W))
}

class DedupQueues(n: Int) extends Module {
  require(n > 0)
  val io = IO(new DedupIO)
  val queues = Seq.fill(n)(Module(new Queue(UInt(32.W), 4)))
  var port = io.in
  for (q <- queues) {
    q.io.enq <> port
    port = q.io.deq
  }
  io.out <> port
}

/* This module creates a Queue in a nested function (such that it is not named via reflection). The
 * default naming for instances prior to #470 caused otherwise identical instantiations of this
 * module to have different instance names for the queues which prevented deduplication.
 * NestedDedup instantiates this module twice to ensure it is deduplicated properly.
 */
class DedupSubModule extends Module {
  val io = IO(new DedupIO)
  io.out <> Queue(io.in, 4)
}

class NestedDedup extends Module {
  val io = IO(new DedupIO)
  val inst0 = Module(new DedupSubModule)
  val inst1 = Module(new DedupSubModule)
  inst0.io.in <> io.in
  inst1.io.in <> inst0.io.out
  io.out <> inst1.io.out
}

object DedupConsts {
  val foo = 3.U
}

class SharedConstantValDedup extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  io.out := io.in + DedupConsts.foo
}

class SharedConstantValDedupTop extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(8.W))
    val out = Output(UInt(8.W))
  })
  val inst0 = Module(new SharedConstantValDedup)
  val inst1 = Module(new SharedConstantValDedup)
  inst0.io.in := io.in
  inst1.io.in := io.in
  io.out := inst0.io.out + inst1.io.out
}


class DedupSpec extends ChiselFlatSpec {
  private val ModuleRegex = """\s*module\s+(\w+)\b.*""".r
  def countModules(verilog: String): Int =
    (verilog split "\n"  collect { case ModuleRegex(name) => name }).size

  "Deduplication" should "occur" in {
    assert(countModules(compile { new DedupQueues(4) }) === 2)
  }

  it should "properly dedup modules with deduped submodules" in {
    assert(countModules(compile { new NestedDedup }) === 3)
  }

  it should "dedup modules that share a literal" in {
    assert(countModules(compile { new SharedConstantValDedupTop  }) === 2)
  }
}

