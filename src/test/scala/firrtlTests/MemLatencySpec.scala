package firrtlTests

import firrtlTests.execution._

object MemLatencySpec {
  case class Write(addr: Int, data: Int, mask: Option[Boolean] = None)
  case class Read(addr: Int, expectedValue: Int)
  case class MemAccess(w: Option[Write], r: Option[Read])
  def writeOnly(addr: Int, data:          Int) = MemAccess(Some(Write(addr, data)), None)
  def readOnly(addr:  Int, expectedValue: Int) = MemAccess(None, Some(Read(addr, expectedValue)))
}

abstract class MemLatencySpec(rLatency: Int, wLatency: Int, ruw: String)
    extends SimpleExecutionTest
    with VerilogExecution {

  import MemLatencySpec._

  require(rLatency >= 0, s"Illegal read-latency ${rLatency} supplied to MemLatencySpec")
  require(wLatency > 0, s"Illegal write-latency ${wLatency} supplied to MemLatencySpec")

  val body =
    s"""mem m :
       |  data-type => UInt<32>
       |  depth => 256
       |  reader => r
       |  writer => w
       |  read-latency => ${rLatency}
       |  write-latency => ${wLatency}
       |  read-under-write => ${ruw}
       |m.r.clk <= clock
       |m.w.clk <= clock
       |""".stripMargin

  val memAccesses: Seq[MemAccess]

  def mask2Poke(m: Option[Boolean]) = m match {
    case Some(false) => Poke("m.w.mask", 0)
    case _           => Poke("m.w.mask", 1)
  }

  def wPokes = memAccesses.map {
    case MemAccess(Some(Write(a, d, m)), _) =>
      Seq(Poke("m.w.en", 1), Poke("m.w.addr", a), Poke("m.w.data", d), mask2Poke(m))
    case _ => Seq(Poke("m.w.en", 0), Invalidate("m.w.addr"), Invalidate("m.w.data"))
  }

  def rPokes = memAccesses.map {
    case MemAccess(_, Some(Read(a, _))) => Seq(Poke("m.r.en", 1), Poke("m.r.addr", a))
    case _                              => Seq(Poke("m.r.en", 0), Invalidate("m.r.addr"))
  }

  // Need to idle for <rLatency> cycles at the end
  val idle = Seq(Poke("m.w.en", 0), Poke("m.r.en", 0))
  def pokes = (wPokes.zip(rPokes)).map { case (wp, rp) => wp ++ rp } ++ Seq.fill(rLatency)(idle)

  // Need to delay read value expects by <rLatency>
  def expects = Seq.fill(rLatency)(Seq(Step(1))) ++ memAccesses.map {
    case MemAccess(_, Some(Read(_, expected))) => Seq(Expect("m.r.data", expected), Step(1))
    case _                                     => Seq(Step(1))
  }

  def commands: Seq[SimpleTestCommand] = (pokes.zip(expects)).flatMap { case (p, e) => p ++ e }
}

trait ToggleMaskAndEnable {
  import MemLatencySpec._

  /**
    * A canonical sequence of memory accesses for sanity checking memories of different latencies.
    * The shortest true "RAW" hazard is reading address 14 two accesses after writing it. Since this
    * access assumed the new value of 87, this means that the access pattern is only valid for
    * certain combinations of read- and write-latencies that vary between read- and write-first
    * memories.
    *
    * @note Read-first mems should return expected values for (write-latency <= 2)
    * @note Write-first mems should return expected values for (write-latency <= read-latency + 2)
    */
  val memAccesses: Seq[MemAccess] = Seq(
    MemAccess(Some(Write(6, 32)), None),
    MemAccess(Some(Write(14, 87)), None),
    MemAccess(None, None),
    MemAccess(Some(Write(19, 63)), Some(Read(14, 87))),
    MemAccess(Some(Write(22, 49)), None),
    MemAccess(Some(Write(11, 99)), Some(Read(6, 32))),
    MemAccess(Some(Write(42, 42)), None),
    MemAccess(Some(Write(77, 81)), None),
    MemAccess(Some(Write(6, 7)), Some(Read(19, 63))),
    MemAccess(Some(Write(39, 5)), Some(Read(42, 42))),
    MemAccess(Some(Write(39, 6, Some(false))), Some(Read(77, 81))), // set mask to zero, should not write
    MemAccess(None, Some(Read(6, 7))), // also read a twice-written address
    MemAccess(None, Some(Read(39, 5))) // ensure masked writes didn't happen
  )
}

/*
 *  This framework is for execution tests, so these tests all focus on
 *  *legal* configurations. Illegal memory parameters that should
 *  result in errors should be tested in MemSpec.
 */

// These two are the same in practice, but the two tests could help expose bugs in VerilogMemDelays
class CombMemSpecNewRUW extends MemLatencySpec(rLatency = 0, wLatency = 1, ruw = "new") with ToggleMaskAndEnable
class CombMemSpecOldRUW extends MemLatencySpec(rLatency = 0, wLatency = 1, ruw = "old") with ToggleMaskAndEnable

// Odd combination: combinational read with 2-cycle write latency
class CombMemWL2SpecNewRUW extends MemLatencySpec(rLatency = 0, wLatency = 2, ruw = "new") with ToggleMaskAndEnable
class CombMemWL2SpecOldRUW extends MemLatencySpec(rLatency = 0, wLatency = 2, ruw = "old") with ToggleMaskAndEnable

// Standard sync read mem
class WriteFirstMemToggleSpec extends MemLatencySpec(rLatency = 1, wLatency = 1, ruw = "new") with ToggleMaskAndEnable
class ReadFirstMemToggleSpec extends MemLatencySpec(rLatency = 1, wLatency = 1, ruw = "old") with ToggleMaskAndEnable

// Read latency 2
class WriteFirstMemToggleSpecRL2
    extends MemLatencySpec(rLatency = 2, wLatency = 1, ruw = "new")
    with ToggleMaskAndEnable
class ReadFirstMemToggleSpecRL2 extends MemLatencySpec(rLatency = 2, wLatency = 1, ruw = "old") with ToggleMaskAndEnable

// Write latency 2
class WriteFirstMemToggleSpecWL2
    extends MemLatencySpec(rLatency = 1, wLatency = 2, ruw = "new")
    with ToggleMaskAndEnable
class ReadFirstMemToggleSpecWL2 extends MemLatencySpec(rLatency = 1, wLatency = 2, ruw = "old") with ToggleMaskAndEnable

// Read latency 2, write latency 2
class WriteFirstMemToggleSpecRL2WL2
    extends MemLatencySpec(rLatency = 2, wLatency = 2, ruw = "new")
    with ToggleMaskAndEnable
class ReadFirstMemToggleSpecRL2WL2
    extends MemLatencySpec(rLatency = 2, wLatency = 2, ruw = "old")
    with ToggleMaskAndEnable

// Read latency 3, write latency 2
class WriteFirstMemToggleSpecRL3WL2
    extends MemLatencySpec(rLatency = 3, wLatency = 2, ruw = "new")
    with ToggleMaskAndEnable
class ReadFirstMemToggleSpecRL3WL2
    extends MemLatencySpec(rLatency = 3, wLatency = 2, ruw = "old")
    with ToggleMaskAndEnable

// Read latency 2, write latency 4 -> ToggleSpec pattern only valid for write-first at this combo
class WriteFirstMemToggleSpecRL2WL4
    extends MemLatencySpec(rLatency = 2, wLatency = 4, ruw = "new")
    with ToggleMaskAndEnable
