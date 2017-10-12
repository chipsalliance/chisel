package chiselTests

import collection.mutable

import chisel3._
import chisel3.util._

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
    expectModuleName(Module(new Queue(UInt(i.W), 16)), genModName("Queue", i))
  }
}

// Note this only checks Iterable[Chisel.Data] which excludes Maps
class IterableNaming extends NamedModuleTester {
  val seq = Seq.tabulate(3) { i =>
    Seq.tabulate(2) { j => expectName(WireInit((i * j).U), s"seq_${i}_${j}") }
  }
  val optSet = Some(Set(expectName(WireInit(0.U), "optSet_0"),
                        expectName(WireInit(1.U), "optSet_1"),
                        expectName(WireInit(2.U), "optSet_2"),
                        expectName(WireInit(3.U), "optSet_3")))

  val stack = mutable.Stack[Module]()
  for (i <- 0 until 4) {
    val j = 3 - i
    stack.push(expectName(Module(new Other(i)), s"stack_$j"))
  }

  def streamFrom(x: Int): Stream[Module] =
    expectName(Module(new Other(x)), s"list_$x") #:: streamFrom(x + 1)
  val stream = streamFrom(0) // Check that we don't get into infinite loop
  val list = stream.take(8).toList
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

  behavior of "Better Naming"

  it should "provide unique counters for each name" in {
    var module: PerNameIndexing = null
    elaborate { module = new PerNameIndexing(4); module }
    assert(module.getNameFailures() == Nil)
  }

  it should "provide names for things defined in Iterable[HasId] and Option[HasId]" in {
    var module: IterableNaming = null
    elaborate { module = new IterableNaming; module }
    assert(module.getNameFailures() == Nil)
  }

  it should "allow digits to be field names in Records" in {
    var module: DigitFieldNamesInRecord  = null
    elaborate { module = new DigitFieldNamesInRecord; module }
    assert(module.getNameFailures() == Nil)
  }
}
