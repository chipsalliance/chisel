// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage
import chisel3.util._

import scala.collection.mutable

// Defined outside of the class so we don't get $ in name
class Other(w: Int) extends Module {
  val io = IO(new Bundle {
    val a = UInt(w.W)
  })
}

// Check the names of the Modules (not instances)
class PerNameIndexing(count: Int) extends NamedModuleTester {
  def genModName(prefix: String, idx: Int): String = if (idx == 0) prefix else s"${prefix}_$idx"
  val wires = Seq.tabulate(count) { i =>
    expectModuleName(Module(new Other(i)), genModName("Other", i))
  }
  val queues = Seq.tabulate(count) { i =>
    expectModuleName(
      Module(new Queue(UInt(i.W), 16) {
        // For this test we need to override desiredName to give the old name, so that indexing
        // is properly tested
        override def desiredName = "Queue"
      }),
      genModName("Queue", i)
    )
  }
}

// Note this only checks Iterable[Chisel.Data] which excludes Maps
class IterableNaming extends NamedModuleTester {
  val seq = Seq.tabulate(3) { i =>
    Seq.tabulate(2) { j => expectName(WireDefault((i * j).U), s"seq_${i}_${j}") }
  }
  val optSet = Some(
    Set(
      expectName(WireDefault(0.U), "optSet_0"),
      expectName(WireDefault(1.U), "optSet_1"),
      expectName(WireDefault(2.U), "optSet_2"),
      expectName(WireDefault(3.U), "optSet_3")
    )
  )

  val stack = {
    val s = mutable.Stack[Module]()
    for (i <- 0 until 4) {
      val j = 3 - i
      s.push(expectName(Module(new Other(i)), s"stack_$j"))
    }
    s
  }
  // Check that we don't get into infinite loop
  // When we still had reflective naming, we could have the list take from the Stream and have
  // everything named list_<n>. Without reflective naming, the first element in the Stream gets a
  // default name because it is built eagerly but the compiler plugin doesn't know how to handle
  // infinite-size structures. Scala 2.13 LazyList would give the same old naming behavior but does
  // not exist in Scala 2.12 so this test has been simplified a bit.
  val stream = LazyList.continually(Module(new Other(8)))
  val list = List.tabulate(4)(i => expectName(Module(new Other(i)), s"list_$i"))
}

class DigitFieldNamesInRecord extends NamedModuleTester {
  val wire = Wire(new CustomBundle("0" -> UInt(32.W), "1" -> UInt(32.W)))
  expectName(wire("0"), "wire.0")
  expectName(wire("1"), "wire.1")
}

/* Better Naming Tests
 *
 * These tests are intended to validate that Chisel picks better names
 */
class BetterNamingTests extends ChiselFlatSpec {

  behavior.of("Better Naming")

  it should "provide unique counters for each name" in {
    var module: PerNameIndexing = null
    ChiselStage.emitCHIRRTL { module = new PerNameIndexing(4); module }
    assert(module.getNameFailures() == Nil)
  }

  it should "provide names for things defined in Iterable[HasId] and Option[HasId]" in {
    var module: IterableNaming = null
    ChiselStage.emitCHIRRTL { module = new IterableNaming; module }
    assert(module.getNameFailures() == Nil)
  }

  it should "allow digits to be field names in Records" in {
    var module: DigitFieldNamesInRecord = null
    ChiselStage.emitCHIRRTL { module = new DigitFieldNamesInRecord; module }
    assert(module.getNameFailures() == Nil)
  }

  "Literals" should "not impact temporary name suffixes" in {
    class MyModule(withLits: Boolean) extends Module {
      val io = IO(new Bundle {})
      if (withLits) {
        List(8.U, -3.S)
      }
      WireDefault(3.U)
    }
    val withLits = ChiselStage.emitCHIRRTL(new MyModule(true))
    val noLits = ChiselStage.emitCHIRRTL(new MyModule(false))
    withLits should equal(noLits)
  }
}
