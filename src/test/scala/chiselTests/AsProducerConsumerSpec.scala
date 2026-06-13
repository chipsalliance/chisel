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
    (log should not).include(warnMsg)
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

  // ======================== Vec ========================

  behavior.of("asProducer/asConsumer with Vec")

  it should "make aligned Vec[Bundle] elements read-only as producer" in {
    // data is aligned → read-only as producer
    checkProducerAlignedError(new RawModule {
      val w = Wire(Vec(2, new MixedBundle))
      val p = w.asProducer
      p(0).data := 1.U
    })
    // ready is flipped → writable as producer
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(Vec(2, new MixedBundle))
      val p = w.asProducer
      p(0).ready := true.B
    })
  }

  it should "make flipped Vec[Bundle] elements read-only as consumer" in {
    // ready is flipped → read-only as consumer
    checkConsumerFlippedError(new RawModule {
      val w = Wire(Vec(2, new MixedBundle))
      val c = w.asConsumer
      c(0).ready := true.B
    })
    // data and valid are aligned → writable as consumer
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(Vec(2, new MixedBundle))
      val c = w.asConsumer
      c(0).data := 1.U
      c(0).valid := true.B
    })
  }

  it should "preserve per-leaf alignment for a Flipped(Vec[Bundle]) viewed as producer" in {
    // Observed behavior: Flipped on a Wire of an aggregate does NOT invert the
    // per-leaf alignment of the resulting hardware (a Wire is its own reference;
    // the outer Flipped on the wire's type is coerced away). So the alignment
    // matches the plain Vec case:
    //   data/valid: aligned → read-only as producer
    //   ready: flipped → writable as producer
    checkProducerAlignedError(new RawModule {
      val w = Wire(Flipped(Vec(2, new MixedBundle)))
      val p = w.asProducer
      p(0).data := 1.U // aligned → read-only as producer
    })
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(Flipped(Vec(2, new MixedBundle)))
      val p = w.asProducer
      p(0).ready := true.B // flipped → writable as producer
    })
  }

  it should "make all elements of a plain Vec[UInt] read-only as producer" in {
    checkProducerAlignedError(new RawModule {
      val w = Wire(Vec(2, UInt(8.W)))
      val p = w.asProducer
      p(0) := 1.U
    })
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
      val b = WireInit(false.B)
      define(w.p, ProbeValue(b))
      val p = w.asProducer
      p.data := 1.U // aligned → read-only as producer
    })
  }

  // ======================== DontCare ========================

  behavior.of("asProducer/asConsumer with DontCare")

  it should "allow assigning DontCare to a writable (flipped) producer-view field" in {
    ChiselStage.emitCHIRRTL(new RawModule {
      val w = Wire(new MixedBundle)
      val p = w.asProducer
      p.ready := DontCare // flipped → writable
    })
  }

  it should "still hard-error when assigning DontCare to an aligned producer-view field" in {
    checkProducerAlignedError(new RawModule {
      val w = Wire(new MixedBundle)
      val p = w.asProducer
      p.data := DontCare // aligned → read-only, even with DontCare RHS
    })
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
    val plain = ChiselStage.emitCHIRRTL(new RawModule {
      override def desiredName = "Plain"
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out :<>= in
    })
    val viewed = ChiselStage.emitCHIRRTL(new RawModule {
      override def desiredName = "Viewed"
      val in = IO(Flipped(new MixedBundle))
      val out = IO(new MixedBundle)
      out.asConsumer :<>= in.asProducer
    })
    // Observed benign difference: the plain form emits a single *bulk* connect
    // ("connect out, in"), while the view form expands it into the explicit
    // per-leaf connects it is equivalent to ("connect out.data, in.data", etc.,
    // with the flipped `ready` connected in the reverse direction). Both are
    // semantically identical. To assert equivalence we normalize each to the set
    // of leaf field names that participate in connects, expanding a bulk
    // root-to-root connect into the full set of MixedBundle leaves.
    val leaves = Set("data", "valid", "ready")
    def connectedLeaves(chirrtl: String): Set[String] =
      chirrtl.linesIterator
        .map(_.trim)
        .filter(_.startsWith("connect "))
        .flatMap { line =>
          val mentioned = leaves.filter(f => line.contains("." + f))
          // A bare root-to-root bulk connect (no field suffixes) covers all leaves.
          if (mentioned.isEmpty) leaves else mentioned
        }
        .toSet
    connectedLeaves(viewed) should be(connectedLeaves(plain))
    // And the plain form is bulk while the viewed form is expanded, confirming
    // the difference is purely structural (bulk vs. inlined per-leaf connects).
    plain should include("connect out, in")
    viewed should include("connect out.data, in.data")
  }

  // ======================== Composition with .readOnly ========================

  behavior.of("composition with .readOnly (regression for stacked writability)")

  it should "hard-error (not warn) when writing through readOnly.asProducerDeprecated" in {
    // The underlying .readOnly is a hard read-only; stacking the deprecated
    // producer view must NOT downgrade it to a warning.
    checkError("Cannot connect to read-only value")(new RawModule {
      val w = Wire(new MixedBundle)
      val p = w.readOnly.asProducerDeprecated
      p.data := 1.U
    })
  }

  it should "not merely warn when writing through readOnly.asProducerDeprecated" in {
    // Confirm it throws rather than only emitting a warning.
    an[Exception] should be thrownBy {
      ChiselStage.emitCHIRRTL(new RawModule {
        val w = Wire(new MixedBundle)
        val p = w.readOnly.asProducerDeprecated
        p.data := 1.U
      })
    }
  }

  it should "hard-error (not warn) when writing a flipped field through readOnly.asConsumerDeprecated" in {
    checkError("Cannot connect to read-only value")(new RawModule {
      val w = Wire(new MixedBundle)
      val c = w.readOnly.asConsumerDeprecated
      c.ready := true.B
    })
  }
}
