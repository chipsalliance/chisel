package chiselTests

import org.scalatest.{FlatSpec, Matchers}
import collection.mutable

import chisel3._
import chisel3.util._

// Defined outside of the class so we don't get $ in name
class Other(w: Int) extends Module {
  val io = new Bundle {
    val a = UInt(w.W)
  }
}
class PerNameIndexing(count: Int) extends Module {
  val io = new Bundle { }

  val wires = Seq.tabulate(count) { i => Module(new Other(i)) }
  val queues = Seq.tabulate(count) { i => Module(new Queue(UInt(i.W), 16)) }
}

// Note this only checks Iterable[Chisel.Data] which excludes Maps
class IterableNaming extends Module {
  val io = new Bundle { }

  val seq = Seq.tabulate(3) { i =>
    Seq.tabulate(2) { j => Wire(init = (i * j).U) }
  }
  val optSet = Some(Set(Wire(init = 0.U),
                        Wire(init = 1.U),
                        Wire(init = 2.U),
                        Wire(init = 3.U)))

  val stack = mutable.Stack[Module]()
  for (i <- 0 until 4) {
    stack push Module(new Other(i))
  }

  def streamFrom(x: Int): Stream[Module] =
    Module(new Other(x)) #:: streamFrom(x + 1)
  val stream = streamFrom(0) // Check that we don't get into infinite loop
  val list = stream.take(8).toList
}

/* Better Naming Tests
 *
 * These tests are intended to validate that Chisel picks better names
 */
class BetterNamingTests extends FlatSpec {

  behavior of "Better Naming"

  it should "provide unique counters for each name" in {
    val verilog = Driver.emit(() => new PerNameIndexing(4))
    val ModuleDef = """\s*module\s+(\S+)\s+:\s*""".r
    val expectedModules = Set("PerNameIndexing",
      "Queue", "Queue_1", "Queue_2", "Queue_3",
      "Other", "Other_1", "Other_2", "Other_3")
    val foundModules = for {
      ModuleDef(name) <- verilog.split("\n").toSeq
    } yield name
    assert(foundModules.toSet === expectedModules)
  }

  it should "provide names for things defined in Iterable[HasId] and Option[HasId]" in {
    val verilog = Driver.emit(() => new IterableNaming)

    val lines = verilog.split("\n").toSeq

    val SeqDef = """\s*wire\s+seq_(\d+)_(\d+)\s+:\s+UInt\s*""".r
    val seqs = for {
      i <- (0 until 3)
      j <- (0 until 2)
    } yield (i.toString, j.toString)
    val foundSeqs = for {
      SeqDef(i, j) <- lines
    } yield (i, j)
    assert(foundSeqs.toSet === seqs.toSet)

    val OptSetDef = """\s*wire\s+optSet_(\d+)\s+:\s+UInt\s*""".r
    val optSets = (0 until 4) map (_.toString)
    val foundOptSets = for {
      OptSetDef(i) <- lines
    } yield i
    assert(foundOptSets.toSet === optSets.toSet)

    val StackDef = """\s*inst\s+stack_(\d+)\s+of\s+Other.*""".r
    val stacks = (0 until 4) map (_.toString)
    val foundStacks = for {
      StackDef(i) <- lines
    } yield i
    assert(foundStacks.toSet === stacks.toSet)

    val ListDef = """\s*inst\s+list_(\d+)\s+of\s+Other.*""".r
    val lists = (0 until 8) map (_.toString)
    val foundLists = for {
      ListDef(i) <- lines
    } yield i
    assert(foundLists.toSet === lists.toSet)
  }
}
