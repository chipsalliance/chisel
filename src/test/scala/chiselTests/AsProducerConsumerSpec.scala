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

  // Wrap `t` in a Wire, apply the given role view, and run `body` on the view. Each builder
  // returns a RawModule; pass it to a check helper or ChiselStage to elaborate it.
  def producerView[T <: Data](t: => T)(body: T => Unit): RawModule = new RawModule { body(Wire(t).asProducer) }
  def consumerView[T <: Data](t: => T)(body: T => Unit): RawModule = new RawModule { body(Wire(t).asConsumer) }
  def producerDeprecatedView[T <: Data](t: => T)(body: T => Unit): RawModule =
    new RawModule { body(Wire(t).asProducerDeprecated) }
  def consumerDeprecatedView[T <: Data](t: => T)(body: T => Unit): RawModule =
    new RawModule { body(Wire(t).asConsumerDeprecated) }

  // Module with an aligned `out` and a Flipped `in` port (both MixedBundle), for connection tests.
  def ioModule(body: (MixedBundle, MixedBundle) => Unit): RawModule = new RawModule {
    val in = IO(Flipped(new MixedBundle))
    val out = IO(new MixedBundle)
    body(out, in)
  }

  // ======================== asProducer per-field writability ========================

  behavior.of("asProducer")

  it should "make aligned fields of a Bundle read-only" in {
    checkProducerAlignedError(producerView(new MixedBundle)(_.data := 1.U))
    checkProducerAlignedError(producerView(new MixedBundle)(_.valid := true.B))
  }

  it should "leave flipped fields of a Bundle writable" in {
    ChiselStage.emitCHIRRTL(producerView(new MixedBundle)(_.ready := true.B))
  }

  it should "make a standalone UInt read-only (aligned with itself)" in {
    checkProducerAlignedError(producerView(UInt(8.W))(_ := 1.U))
  }

  it should "work correctly with DecoupledIO" in {
    // bits and valid are aligned (read-only as producer), ready is flipped (writable)
    checkProducerAlignedError(producerView(new DecoupledIO(UInt(8.W)))(_.bits := 1.U))
    checkProducerAlignedError(producerView(new DecoupledIO(UInt(8.W)))(_.valid := true.B))
    ChiselStage.emitCHIRRTL(producerView(new DecoupledIO(UInt(8.W)))(_.ready := true.B))
  }

  it should "handle nested bundles correctly" in {
    // inner.ready: flipped within aligned → flipped → writable
    ChiselStage.emitCHIRRTL(producerView(new NestedBundle)(_.inner.ready := true.B))
    // flippedInner.data: aligned within flipped → flipped → writable
    ChiselStage.emitCHIRRTL(producerView(new NestedBundle)(_.flippedInner.data := 1.U))
    // flippedInner.ready: flipped within flipped → aligned → read-only
    checkProducerAlignedError(producerView(new NestedBundle)(_.flippedInner.ready := true.B))
  }

  it should "handle coerced (Input/Output) fields correctly" in {
    // Output coerces all children to aligned → read-only
    checkProducerAlignedError(producerView(new CoercedBundle)(_.out.data := 1.U))
    checkProducerAlignedError(producerView(new CoercedBundle)(_.out.ready := true.B))
    // Input coerces all children to flipped → writable
    ChiselStage.emitCHIRRTL(producerView(new CoercedBundle)(_.in.data := 1.U))
    ChiselStage.emitCHIRRTL(producerView(new CoercedBundle)(_.in.ready := true.B))
  }

  it should "work correctly on RHS of :<>=" in {
    ChiselStage.emitCHIRRTL(ioModule((out, in) => out :<>= in.asProducer))
  }

  it should "work correctly on RHS of :<=" in {
    ChiselStage.emitCHIRRTL(ioModule((out, in) => out :<= in.asProducer))
  }

  it should "work correctly on RHS of :>=" in {
    ChiselStage.emitCHIRRTL(ioModule((out, in) => out :>= in.asProducer))
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
    checkConsumerFlippedError(consumerView(new MixedBundle)(_.ready := true.B))
  }

  it should "leave aligned fields of a Bundle writable" in {
    ChiselStage.emitCHIRRTL(consumerView(new MixedBundle) { c =>
      c.data := 1.U
      c.valid := true.B
    })
  }

  it should "leave a standalone UInt writable (no flipped fields)" in {
    ChiselStage.emitCHIRRTL(consumerView(UInt(8.W))(_ := 1.U))
  }

  it should "work correctly with DecoupledIO" in {
    // ready is flipped (read-only as consumer), bits and valid are aligned (writable)
    checkConsumerFlippedError(consumerView(new DecoupledIO(UInt(8.W)))(_.ready := true.B))
    ChiselStage.emitCHIRRTL(consumerView(new DecoupledIO(UInt(8.W))) { c =>
      c.bits := 1.U
      c.valid := true.B
    })
  }

  it should "handle nested bundles correctly" in {
    // flippedInner.ready: flipped within flipped → aligned → writable
    ChiselStage.emitCHIRRTL(consumerView(new NestedBundle)(_.flippedInner.ready := true.B))
    // inner.ready: flipped within aligned → flipped → read-only
    checkConsumerFlippedError(consumerView(new NestedBundle)(_.inner.ready := true.B))
  }

  it should "work correctly on LHS of :<>=" in {
    ChiselStage.emitCHIRRTL(ioModule((out, in) => out.asConsumer :<>= in))
  }

  it should "work correctly on LHS of :<=" in {
    ChiselStage.emitCHIRRTL(ioModule((out, in) => out.asConsumer :<= in))
  }

  it should "work correctly on LHS of :>=" in {
    ChiselStage.emitCHIRRTL(ioModule((out, in) => out.asConsumer :>= in))
  }

  it should "NOT create a view for literals" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = 123.U
      assert(a.asConsumer eq a)
    })
  }

  it should "NOT create a view for op results" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val a = IO(Input(UInt(8.W)))
      val x = a + 1.U
      assert(x.asConsumer eq x)
    })
  }

  // ======================== Side enforcement ========================

  behavior.of("asProducer side enforcement")

  it should "error when asProducer is used on LHS of :<>=" in {
    checkProducerOnLHSError(ioModule((out, in) => out.asProducer :<>= in))
  }

  it should "error when asProducer is used on LHS of :<=" in {
    checkProducerOnLHSError(ioModule((out, in) => out.asProducer :<= in))
  }

  it should "error when asProducer is used on LHS of :>=" in {
    checkProducerOnLHSError(ioModule((out, in) => out.asProducer :>= in))
  }

  it should "error when asProducer is used on LHS of :#=" in {
    checkProducerOnLHSError(ioModule((out, in) => out.asProducer :#= in))
  }

  behavior.of("asConsumer side enforcement")

  it should "error when asConsumer is used on RHS of :<>=" in {
    checkConsumerOnRHSError(ioModule((out, in) => out :<>= in.asConsumer))
  }

  it should "error when asConsumer is used on RHS of :<=" in {
    checkConsumerOnRHSError(ioModule((out, in) => out :<= in.asConsumer))
  }

  it should "error when asConsumer is used on RHS of :>=" in {
    checkConsumerOnRHSError(ioModule((out, in) => out :>= in.asConsumer))
  }

  it should "error when asConsumer is used on RHS of :#=" in {
    checkConsumerOnRHSError(ioModule((out, in) => out :#= in.asConsumer))
  }

  // ======================== Combined usage ========================

  behavior.of("asProducer and asConsumer together")

  it should "work when both are used correctly" in {
    ChiselStage.emitCHIRRTL(ioModule((out, in) => out.asConsumer :<>= in.asProducer))
  }

  // ======================== Deprecated variants ========================

  def checkHasWarning(warnMsg: String)(m: => RawModule)(implicit pos: Position): Unit = {
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(m))
    log should include(warnMsg)
  }

  def checkNoWarning(warnMsg: String)(m: => RawModule)(implicit pos: Position): Unit = {
    val (log, _) = grabLog(ChiselStage.emitCHIRRTL(m))
    (log should not).include(warnMsg)
  }

  // The per-operator side-enforcement matrix is covered by the hard variants above; the
  // deprecated variants only spot-check :<>= since they share the same enforcement path.
  behavior.of("asProducerDeprecated")

  it should "warn (not error) when connecting to aligned fields" in {
    checkHasWarning("Cannot connect to producer's aligned field")(
      producerDeprecatedView(new MixedBundle)(_.data := 1.U)
    )
  }

  it should "leave flipped fields writable without warning" in {
    checkNoWarning("producer")(producerDeprecatedView(new MixedBundle)(_.ready := true.B))
  }

  it should "warn (not error) when used on LHS" in {
    checkHasWarning(".asProducer cannot be used on the consumer (LHS)")(
      ioModule((out, in) => out.asProducerDeprecated :<>= in)
    )
  }

  it should "work correctly on RHS without warning" in {
    checkNoWarning("producer")(ioModule((out, in) => out :<>= in.asProducerDeprecated))
  }

  behavior.of("asConsumerDeprecated")

  it should "warn (not error) when connecting to flipped fields" in {
    checkHasWarning("Cannot connect to consumer's flipped field")(
      consumerDeprecatedView(new MixedBundle)(_.ready := true.B)
    )
  }

  it should "leave aligned fields writable without warning" in {
    checkNoWarning("consumer")(consumerDeprecatedView(new MixedBundle)(_.data := 1.U))
  }

  it should "warn (not error) when used on RHS" in {
    checkHasWarning(".asConsumer cannot be used on the producer (RHS)")(
      ioModule((out, in) => out :<>= in.asConsumerDeprecated)
    )
  }

  it should "work correctly on LHS without warning" in {
    checkNoWarning("consumer")(ioModule((out, in) => out.asConsumerDeprecated :<>= in))
  }

  // ======================== Vec ========================

  behavior.of("asProducer/asConsumer with Vec")

  it should "make aligned Vec[Bundle] elements read-only as producer" in {
    // data is aligned → read-only as producer
    checkProducerAlignedError(producerView(Vec(2, new MixedBundle))(v => v(0).data := 1.U))
    // ready is flipped → writable as producer
    ChiselStage.emitCHIRRTL(producerView(Vec(2, new MixedBundle))(v => v(0).ready := true.B))
  }

  it should "make flipped Vec[Bundle] elements read-only as consumer" in {
    // ready is flipped → read-only as consumer
    checkConsumerFlippedError(consumerView(Vec(2, new MixedBundle))(v => v(0).ready := true.B))
    // data and valid are aligned → writable as consumer
    ChiselStage.emitCHIRRTL(consumerView(Vec(2, new MixedBundle)) { v =>
      v(0).data := 1.U
      v(0).valid := true.B
    })
  }

  it should "preserve per-leaf alignment for a Flipped(Vec[Bundle]) viewed as producer" in {
    // Observed behavior: Flipped on a Wire of an aggregate does NOT invert the per-leaf
    // alignment of the resulting hardware (a Wire is its own reference; the outer Flipped on
    // the wire's type is coerced away), so the alignment matches the plain Vec case.
    checkProducerAlignedError(producerView(Flipped(Vec(2, new MixedBundle)))(v => v(0).data := 1.U))
    ChiselStage.emitCHIRRTL(producerView(Flipped(Vec(2, new MixedBundle)))(v => v(0).ready := true.B))
  }

  it should "make all elements of a plain Vec[UInt] read-only as producer" in {
    checkProducerAlignedError(producerView(Vec(2, UInt(8.W)))(v => v(0) := 1.U))
  }

  // ======================== Probe ========================

  behavior.of("asProducer/asConsumer with Probe")

  class ProbeBundle extends Bundle {
    val data = UInt(8.W)
    val p = Probe(Bool())
  }

  it should "view a probe-containing bundle as producer and keep aligned fields read-only" in {
    checkProducerAlignedError(new RawModule {
      val w = Wire(new ProbeBundle)
      // Make the underlying wire legal by defining the probe.
      define(w.p, ProbeValue(WireInit(false.B)))
      w.asProducer.data := 1.U // aligned → read-only as producer
    })
  }

  // ======================== DontCare ========================

  behavior.of("asProducer/asConsumer with DontCare")

  it should "allow assigning DontCare to a writable (flipped) producer-view field" in {
    ChiselStage.emitCHIRRTL(producerView(new MixedBundle)(_.ready := DontCare)) // flipped → writable
  }

  it should "still hard-error when assigning DontCare to an aligned producer-view field" in {
    checkProducerAlignedError(producerView(new MixedBundle)(_.data := DontCare)) // aligned → read-only
  }

  it should "allow plain assignment of DontCare to an unviewed wire" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val out = Wire(new MixedBundle)
      out := DontCare
    })
  }

  // ======================== CHIRRTL equivalence ========================

  behavior.of("CHIRRTL equivalence")

  it should "connect the same leaf fields for out.asConsumer :<>= in.asProducer as out :<>= in" in {
    // The plain form emits a single bulk connect ("connect out, in"); the view form expands
    // it into the equivalent explicit per-leaf connects ("connect out.data, in.data", etc.,
    // with the flipped `ready` reversed). Both are semantically identical; the only difference
    // is structural (bulk vs. inlined per-leaf connects).
    val plain = ChiselStage.emitCHIRRTL(ioModule((out, in) => out :<>= in))
    val viewed = ChiselStage.emitCHIRRTL(ioModule((out, in) => out.asConsumer :<>= in.asProducer))
    plain should include("connect out, in")
    viewed should include("connect out.data, in.data")
  }

  // ======================== Composition with .readOnly ========================

  behavior.of("composition with .readOnly (regression for stacked writability)")

  it should "hard-error (not warn) when writing through readOnly.asProducerDeprecated" in {
    // The underlying .readOnly is a hard read-only; stacking the deprecated producer view
    // must NOT downgrade it to a warning.
    checkError("Cannot connect to read-only value")(new RawModule {
      Wire(new MixedBundle).readOnly.asProducerDeprecated.data := 1.U
    })
  }

  it should "hard-error (not warn) when writing a flipped field through readOnly.asConsumerDeprecated" in {
    checkError("Cannot connect to read-only value")(new RawModule {
      Wire(new MixedBundle).readOnly.asConsumerDeprecated.ready := true.B
    })
  }
}
