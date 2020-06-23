// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.{Queue, EnqIO, DeqIO, QueueIO, log2Ceil}
import chisel3.experimental.{CloneModuleAsRecord, IO}
import chisel3.testers.BasicTester

class MultiIOQueue[T <: Data](gen: T, val entries: Int) extends MultiIOModule {
  val clk = IO(Input(Clock()))
  val rst = IO(Input(Reset()))
  val enq = IO(Flipped(EnqIO(gen)))
  val deq = IO(Flipped(DeqIO(gen)))
  val count = IO(Output(UInt(log2Ceil(entries + 1).W)))
  val impl = withClockAndReset(clk, rst) { Module(new Queue(gen, entries)) }
  impl.io.enq <> enq
  deq <> impl.io.deq
  count := impl.io.count
}

class QueueClone(multiIO: Boolean = false) extends Module {
  val io = IO(new QueueIO(UInt(32.W), 4))
  if (multiIO) {
    val q1 = Module(new MultiIOQueue(UInt(32.W), 2))
    val q2_io = CloneModuleAsRecord(q1)
    q1.clk := clock
    q1.rst := reset
    q1.enq <> io.enq
    q2_io("clk").asInstanceOf[Clock] := clock
    q2_io("rst").asInstanceOf[Reset] := reset
    q2_io("enq").asInstanceOf[q1.enq.type] <> q1.deq
    io.deq <> q2_io("deq").asInstanceOf[q1.deq.type]
    io.count := q1.count + q2_io("count").asInstanceOf[q1.count.type]
  } else {
    val q1 = Module(new Queue(UInt(32.W), 2))
    val q2_io = CloneModuleAsRecord(q1)
    q1.io.enq <> io.enq
    val q2_io_bundle = q2_io("io").asInstanceOf[q1.io.type]
    q2_io_bundle.enq <> q1.io.deq
    io.deq <> q2_io_bundle.deq
    io.count := q1.io.count + q2_io_bundle.count
  }
}

class QueueCloneTester(x: Int, multiIO: Boolean = false) extends BasicTester {
  val dut = Module(new QueueClone(multiIO))
  val start = RegNext(dut.io.enq.fire, true.B)
  val accept = RegNext(dut.io.deq.valid, false.B)
  dut.io.enq.bits := x.U
  dut.io.enq.valid := start
  dut.io.deq.ready := accept
  when (dut.io.deq.fire) {
    assert(dut.io.deq.bits === x.U)
    stop()
  }
}

class CloneModuleSpec extends ChiselPropSpec {

  val xVals = Table(
    ("x"),  // First tuple defines column names
    (42),   // Subsequent tuples define the data
    (63),
    (99))

  property("QueueCloneTester should return the correct result") {
    forAll (xVals) { (x: Int) =>
      assertTesterPasses{ new QueueCloneTester(x) }
    }
  }

  property("QueueClone's cloned queues should share the same module") {
    val c = ChiselStage.convert(new QueueClone)
    assert(c.modules.length == 2)
  }

  property("Clone of MultiIOModule should simulate correctly") {
    forAll (xVals) { (x: Int) =>
      assertTesterPasses{ new QueueCloneTester(x, multiIO=true) }
    }
  }

  property("Clones of MultiIOModules should share the same module") {
    val c = ChiselStage.convert(new QueueClone(multiIO=true))
    assert(c.modules.length == 3)
  }

}
