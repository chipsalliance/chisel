// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.probe._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage
import org.scalactic.source.Position
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class AsProducerConsumerSpec extends AnyFlatSpec with Matchers with LogUtils {

  class MixedBundle extends Bundle {
    val data = UInt(8.W)
    val valid = Bool()
    val ready = Flipped(Bool())
  }

  class NestedBundle extends Bundle {
    val inner = new MixedBundle
    val flippedInner = Flipped(new MixedBundle)
  }

  class CoercedBundle extends Bundle {
    val out = Output(new MixedBundle)
    val in = Input(new MixedBundle)
  }

  def checkError(errMsg: String)(m: => RawModule)(implicit pos: Position): Unit = {
    val e = the[ChiselException] thrownBy {
      ChiselStage.elaborate(m, Array("--throw-on-first-error"))
    }
    e.getMessage should include(errMsg)
  }

  def checkProducerAlignedError(m: => RawModule)(implicit pos: Position): Unit =
    checkError("Cannot connect to producer's aligned field")(m)

  def checkConsumerFlippedError(m: => RawModule)(implicit pos: Position): Unit =
    checkError("Cannot connect to consumer's flipped field")(m)

  def checkProducerOnLHSError(m: => RawModule)(implicit pos: Position): Unit =
    checkError(".asProducer cannot be used on the consumer (LHS)")(m)

  def checkConsumerOnRHSError(m: => RawModule)(implicit pos: Position): Unit =
    checkError(".asConsumer cannot be used on the producer (RHS)")(m)

  // ======================== asProducer per-field writability ========================

  behavior.of("asProducer")

  it should "make aligned fields of a Bundle read-only" in {
    checkProducerAlignedError(new RawModule {
      val w = Wire(new MixedBundle)
      val p = w.asProducer
      p.data := 1.U
    })
    checkProducerAlignedError(new RawModule {
      val w = Wire(new MixedBundle)
      val p = w.asProducer
      p.valid := true.B
    })
  }

  it should "leave flipped fields of a Bundle writable" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new MixedBundle)
      val p = w.asProducer
      p.ready := true.B
    })
  }

  it should "make a standalone UInt read-only (aligned with itself)" in {
    checkProducerAlignedError(new RawModule {
      val w = Wire(UInt(8.W))
      w.asProducer := 1.U
    })
  }

  it should "work correctly with DecoupledIO" in {
    // bits and valid are aligned (read-only as producer), ready is flipped (writable)
    checkProducerAlignedError(new RawModule {
      val w = Wire(new DecoupledIO(UInt(8.W)))
      val p = w.asProducer
      p.bits := 1.U
    })
    checkProducerAlignedError(new RawModule {
      val w = Wire(new DecoupledIO(UInt(8.W)))
      val p = w.asProducer
      p.valid := true.B
    })
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new DecoupledIO(UInt(8.W)))
      val p = w.asProducer
      p.ready := true.B
    })
  }

  it should "handle nested bundles correctly" in {
    // inner is aligned, inner.ready is flipped within aligned → flipped overall
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new NestedBundle)
      val p = w.asProducer
      p.inner.ready := true.B // flipped within aligned → flipped → writable
    })
    // flippedInner is flipped, flippedInner.data is aligned within flipped → flipped overall
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new NestedBundle)
      val p = w.asProducer
      p.flippedInner.data := 1.U // aligned within flipped → flipped → writable
    })
    // flippedInner is flipped, flippedInner.ready is flipped within flipped → aligned overall
    checkProducerAlignedError(new RawModule {
      val w = Wire(new NestedBundle)
      val p = w.asProducer
      p.flippedInner.ready := true.B // flipped within flipped → aligned → read-only
    })
  }

  it should "handle coerced (Input/Output) fields correctly" in {
    // Output coerces all children to aligned
    checkProducerAlignedError(new RawModule {
      val w = Wire(new CoercedBundle)
      val p = w.asProducer
      p.out.data := 1.U // aligned under Output → aligned → read-only
    })
    checkProducerAlignedError(new RawModule {
      val w = Wire(new CoercedBundle)
      val p = w.asProducer
      p.out.ready := true.B // flipped under Output → coerced aligned → read-only
    })
    // Input coerces all children to flipped
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new CoercedBundle)
      val p = w.asProducer
      p.in.data := 1.U // aligned under Input → coerced flipped → writable
    })
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new CoercedBundle)
      val p = w.asProducer
      p.in.ready := true.B // flipped under Input → coerced flipped → writable
    })
  }

  it should "work correctly on RHS of :<>=" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :<>= in.asProducer
    })
  }

  it should "work correctly on RHS of :<=" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :<= in.asProducer
    })
  }

  it should "work correctly on RHS of :>=" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :>= in.asProducer
    })
  }

  it should "NOT create a view for literals" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = 123.U
      assert(a.asProducer eq a)
    })
  }

  it should "NOT create a view for op results" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(UInt(8.W)))
      val x = a + 1.U
      assert(x.asProducer eq x)
    })
  }

  // ======================== asConsumer per-field writability ========================

  behavior.of("asConsumer")

  it should "make flipped fields of a Bundle read-only" in {
    checkConsumerFlippedError(new RawModule {
      val w = Wire(new MixedBundle)
      val c = w.asConsumer
      c.ready := true.B
    })
  }

  it should "leave aligned fields of a Bundle writable" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new MixedBundle)
      val c = w.asConsumer
      c.data := 1.U
      c.valid := true.B
    })
  }

  it should "leave a standalone UInt writable (no flipped fields)" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(UInt(8.W))
      w.asConsumer := 1.U
    })
  }

  it should "work correctly with DecoupledIO" in {
    // ready is flipped (read-only as consumer), bits and valid are aligned (writable)
    checkConsumerFlippedError(new RawModule {
      val w = Wire(new DecoupledIO(UInt(8.W)))
      val c = w.asConsumer
      c.ready := true.B
    })
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new DecoupledIO(UInt(8.W)))
      val c = w.asConsumer
      c.bits := 1.U
      c.valid := true.B
    })
  }

  it should "handle nested bundles correctly" in {
    // flippedInner.ready is flipped within flipped → aligned → writable
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new NestedBundle)
      val c = w.asConsumer
      c.flippedInner.ready := true.B
    })
    // inner.ready is flipped within aligned → flipped → read-only
    checkConsumerFlippedError(new RawModule {
      val w = Wire(new NestedBundle)
      val c = w.asConsumer
      c.inner.ready := true.B
    })
  }

  it should "work correctly on LHS of :<>=" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asConsumer :<>= in
    })
  }

  it should "work correctly on LHS of :<=" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asConsumer :<= in
    })
  }

  it should "work correctly on LHS of :>=" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asConsumer :>= in
    })
  }

  it should "NOT create a view for literals" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = 123.U
      assert(a.asConsumer eq a)
    })
  }

  // ======================== Side enforcement ========================

  behavior.of("asProducer side enforcement")

  it should "error when asProducer is used on LHS of :<>=" in {
    checkProducerOnLHSError(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asProducer :<>= in
    })
  }

  it should "error when asProducer is used on LHS of :<=" in {
    checkProducerOnLHSError(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asProducer :<= in
    })
  }

  it should "error when asProducer is used on LHS of :>=" in {
    checkProducerOnLHSError(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asProducer :>= in
    })
  }

  it should "error when asProducer is used on LHS of :#=" in {
    checkProducerOnLHSError(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asProducer :#= in
    })
  }

  behavior.of("asConsumer side enforcement")

  it should "error when asConsumer is used on RHS of :<>=" in {
    checkConsumerOnRHSError(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :<>= in.asConsumer
    })
  }

  it should "error when asConsumer is used on RHS of :<=" in {
    checkConsumerOnRHSError(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :<= in.asConsumer
    })
  }

  it should "error when asConsumer is used on RHS of :>=" in {
    checkConsumerOnRHSError(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :>= in.asConsumer
    })
  }

  it should "error when asConsumer is used on RHS of :#=" in {
    checkConsumerOnRHSError(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :#= in.asConsumer
    })
  }

  // ======================== Combined usage ========================

  behavior.of("asProducer and asConsumer together")

  it should "work when both are used correctly" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asConsumer :<>= in.asProducer
    })
  }

  // ======================== Deprecated variants ========================

  def checkHasWarning(warnMsg: String)(m: => RawModule)(implicit pos: Position): Unit = {
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(m))
    log should include(warnMsg)
  }

  def checkNoWarning(warnMsg: String)(m: => RawModule)(implicit pos: Position): Unit = {
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(m))
    log should not include (warnMsg)
  }

  behavior.of("asProducerDeprecated")

  it should "warn (not error) when connecting to aligned fields" in {
    checkHasWarning("Cannot connect to producer's aligned field")(new RawModule {
      val w = Wire(new MixedBundle)
      val p = w.asProducerDeprecated
      p.data := 1.U
    })
  }

  it should "leave flipped fields writable without warning" in {
    checkNoWarning("producer")(new RawModule {
      val w = Wire(new MixedBundle)
      val p = w.asProducerDeprecated
      p.ready := true.B
    })
  }

  it should "warn (not error) when used on LHS" in {
    checkHasWarning(".asProducer cannot be used on the consumer (LHS)")(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asProducerDeprecated :<>= in
    })
  }

  it should "work correctly on RHS without warning" in {
    checkNoWarning("producer")(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :<>= in.asProducerDeprecated
    })
  }

  behavior.of("asConsumerDeprecated")

  it should "warn (not error) when connecting to flipped fields" in {
    checkHasWarning("Cannot connect to consumer's flipped field")(new RawModule {
      val w = Wire(new MixedBundle)
      val c = w.asConsumerDeprecated
      c.ready := true.B
    })
  }

  it should "leave aligned fields writable without warning" in {
    checkNoWarning("consumer")(new RawModule {
      val w = Wire(new MixedBundle)
      val c = w.asConsumerDeprecated
      c.data := 1.U
    })
  }

  it should "warn (not error) when used on RHS" in {
    checkHasWarning(".asConsumer cannot be used on the producer (RHS)")(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :<>= in.asConsumerDeprecated
    })
  }

  it should "work correctly on LHS without warning" in {
    checkNoWarning("consumer")(new RawModule {
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asConsumerDeprecated :<>= in
    })
  }
}
