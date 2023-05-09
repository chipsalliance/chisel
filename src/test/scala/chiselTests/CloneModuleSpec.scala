// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.util.{log2Ceil, Decoupled, DeqIO, EnqIO, Queue, QueueIO}
import chisel3.experimental.CloneModuleAsRecord
import chisel3.testers.BasicTester

class MultiIOQueue[T <: Data](gen: T, val entries: Int) extends Module {
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
  when(dut.io.deq.fire) {
    assert(dut.io.deq.bits === x.U)
    stop()
  }
}

class CloneModuleAsRecordAnnotate extends Module {
  override def desiredName = "Top"
  val in = IO(Flipped(Decoupled(UInt(8.W))))
  val out = IO(Decoupled(UInt(8.W)))

  val q1 = Module(new Queue(UInt(8.W), 4))
  val q2 = CloneModuleAsRecord(q1)
  val q2_io = q2("io").asInstanceOf[q1.io.type]
  // Also make a wire to check that cloning works, can be connected to, and annotated
  val q2_wire = {
    val w = Wire(chiselTypeOf(q2))
    w <> q2
    w
  }
  // But connect to the original (using last connect semantics to override connects to wire
  q1.io.enq <> in
  q2_io.enq <> q1.io.deq
  out <> q2_io.deq
}

class CloneModuleSpec extends ChiselPropSpec {

  val xVals = Table(
    ("x"), // First tuple defines column names
    (42), // Subsequent tuples define the data
    (63),
    (99)
  )

  property("QueueCloneTester should return the correct result") {
    forAll(xVals) { (x: Int) =>
      assertTesterPasses { new QueueCloneTester(x) }
    }
  }

  property("QueueClone's cloned queues should share the same module") {
    val c = ChiselStage.convert(new QueueClone)
    assert(c.modules.length == 2)
  }

  property("Clone of Module should simulate correctly") {
    forAll(xVals) { (x: Int) =>
      assertTesterPasses { new QueueCloneTester(x, multiIO = true) }
    }
  }

  property("Clones of Modules should share the same module") {
    val c = ChiselStage.convert(new QueueClone(multiIO = true))
    assert(c.modules.length == 3)
  }

  property("Cloned Modules should annotate correctly") {
    // Hackily get the actually Module object out
    var mod: CloneModuleAsRecordAnnotate = null
    val res = ChiselStage.convert {
      mod = new CloneModuleAsRecordAnnotate
      mod
    }
    // ********** Checking the output of CloneModuleAsRecord **********
    // Note that we overrode desiredName so that Top is named "Top"
    mod.q1.io.enq.toTarget.serialize should be("~Top|Queue4_UInt8>io.enq")
    mod.q2_io.deq.toTarget.serialize should be("~Top|Queue4_UInt8>io.deq")
    mod.q1.io.enq.toAbsoluteTarget.serialize should be("~Top|Top/q1:Queue4_UInt8>io.enq")
    mod.q2_io.deq.toAbsoluteTarget.serialize should be("~Top|Top/q2:Queue4_UInt8>io.deq")
    // Legacy APIs that nevertheless were tricky to get right
    mod.q1.io.enq.toNamed.serialize should be("Top.Queue4_UInt8.io.enq")
    mod.q2_io.deq.toNamed.serialize should be("Top.Queue4_UInt8.io.deq")
    mod.q1.io.enq.instanceName should be("io.enq")
    mod.q2_io.deq.instanceName should be("io.deq")
    mod.q1.io.enq.pathName should be("Top.q1.io.enq")
    mod.q2_io.deq.pathName should be("Top.q2.io.deq")
    mod.q1.io.enq.parentPathName should be("Top.q1")
    mod.q2_io.deq.parentPathName should be("Top.q2")
    mod.q1.io.enq.parentModName should be("Queue4_UInt8")
    mod.q2_io.deq.parentModName should be("Queue4_UInt8")

    // ********** Checking the wire cloned from the output of CloneModuleAsRecord **********
    val wire_io = mod.q2_wire("io").asInstanceOf[QueueIO[UInt]]
    mod.q2_wire.toTarget.serialize should be("~Top|Top>q2_wire")
    wire_io.enq.toTarget.serialize should be("~Top|Top>q2_wire.io.enq")
    mod.q2_wire.toAbsoluteTarget.serialize should be("~Top|Top>q2_wire")
    wire_io.enq.toAbsoluteTarget.serialize should be("~Top|Top>q2_wire.io.enq")
    // Legacy APIs
    mod.q2_wire.toNamed.serialize should be("Top.Top.q2_wire")
    wire_io.enq.toNamed.serialize should be("Top.Top.q2_wire.io.enq")
    mod.q2_wire.instanceName should be("q2_wire")
    wire_io.enq.instanceName should be("q2_wire.io.enq")
    mod.q2_wire.pathName should be("Top.q2_wire")
    wire_io.enq.pathName should be("Top.q2_wire.io.enq")
    mod.q2_wire.parentPathName should be("Top")
    wire_io.enq.parentPathName should be("Top")
    mod.q2_wire.parentModName should be("Top")
    wire_io.enq.parentModName should be("Top")
  }

}
