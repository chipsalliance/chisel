// See LICENSE for license details.

package chisel3.tester

import chisel3._
import chisel3.experimental.{DataMirror, MultiIOModule}
import java.util.concurrent.{Semaphore, ConcurrentLinkedQueue, TimeUnit}
import scala.collection.mutable

import scala.collection.mutable
import firrtl_interpreter._
import treadle.{HasTreadleSuite, TreadleTester}

class TreadleBackend[T <: MultiIOModule](dut: T, tester: TreadleTester)
    extends BackendInstance[T] with ThreadedBackend {

  def getModule: T = dut

  /** Returns a Seq of (data reference, fully qualified element names) for the input.
    * name is the name of data
    */
  protected def getDataNames(name: String, data: Data): Seq[(Data, String)] = Seq(data -> name) ++ (data match {
    case _: Element => Seq()
    case b: Record => b.elements.toSeq flatMap {case (n, e) => getDataNames(s"${name}_$n", e)}
    case v: Vec[_] => v.zipWithIndex flatMap {case (e, i) => getDataNames(s"${name}_$i", e)}
  })

  // TODO: the naming facility should be part of infrastructure not backend
  protected val portNames: Map[Data, String] = DataMirror.modulePorts(dut).flatMap { case (name, data) =>
    getDataNames(name, data).toList
  }.toMap

  protected def resolveName(signal: Data): String = {
    portNames.getOrElse(signal, signal.toString)
  }

  override def pokeBits(signal: Bits, value: BigInt, priority: Int): Unit = {
    if (threadingChecker.doPoke(currentThread.get, signal, value, priority, new Throwable)) {
      // println(s"${portNames(signal)} <- $value")  // TODO: toggle-able debug mode
      tester.poke(portNames(signal), value)
    }
  }

  override def peekBits(signal: Bits, stale: Boolean): BigInt = {
    require(!stale, "Stale peek not yet implemented")

    // TODO: properly determine clock
    threadingChecker.doPeek(currentThread.get, signal, dut.clock, new Throwable)
    val a = tester.peek(portNames(signal))
    // println(s"${portNames(signal)} -> $a")  // TODO: toggle-able debug mode
    a
  }

  override def expectBits(signal: Bits, value: BigInt, stale: Boolean): Unit = {
    require(!stale, "Stale peek not yet implemented")

    // println(s"${portNames(signal)} ?> $value")  // TODO: toggle-able debug mode
    Context().env.testerExpect(value, peekBits(signal, stale), resolveName(signal), None)
  }

  protected val clockCounter : mutable.HashMap[Clock, Int] = mutable.HashMap()
  protected def getClockCycle(clk: Clock): Int = {
    clockCounter.getOrElse(clk, 0)
  }
  protected def getClock(clk: Clock): Boolean = tester.peek(portNames(clk)).toInt match {
    case 0 => false
    case 1 => true
  }

  protected val lastClockValue: mutable.HashMap[Clock, Boolean] = mutable.HashMap()
  protected val threadingChecker = new ThreadingChecker()

  override def timescope(contents: => Unit): Unit = {
    val newTimescope = threadingChecker.newTimescope(currentThread.get)
    contents
    threadingChecker.closeTimescope(newTimescope).foreach { case (data, valueOption) =>
      valueOption match {
        case Some(value) => tester.poke(portNames(data), value)
          // println(s"${portNames(data)} <- (revert) $value")  // TODO: toggle-able debug mode
        case None => tester.poke(portNames(data), 0)  // TODO: randomize or 4-state sim
          // println(s"${portNames(data)} <- (revert) DC")  // TODO: toggle-able debug mode
      }
    }
  }

  override def step(signal: Clock, cycles: Int): Unit = {
    // TODO: maybe a fast condition for when threading is not in use?
    for (_ <- 0 until cycles) {
      // If a new clock, record the current value so change detection is instantaneous
      if (signal != dut.clock && !lastClockValue.contains(signal)) {
        lastClockValue.put(signal, getClock(signal))
      }

      val thisThread = currentThread.get
      blockedThreads.put(signal, blockedThreads.getOrElseUpdate(signal, Seq()) :+ thisThread)
      scheduler()
      thisThread.waiting.acquire()
    }
  }

  protected val interruptedException = new ConcurrentLinkedQueue[Throwable]()

  protected def onException(e: Throwable) {
    interruptedException.offer(e)
  }

  override def run(testFn: T => Unit): Unit = {
    val mainThread = fork {
      tester.poke("reset", 1)
      tester.step(1)
      tester.poke("reset", 0)

      testFn(dut)
    }
    // TODO: stop abstraction-breaking activeThreads
    require(activeThreads.length == 1)  // only thread should be main
    activeThreads.trimStart(1)  // delete active threads - TODO fix this
    blockedThreads.put(dut.clock, Seq(mainThread))  // TODO dehackify, this allows everything below to kick off

    while (!mainThread.done) {  // iterate timesteps
      val unblockedThreads = new mutable.ArrayBuffer[TesterThread]()

      // Unblock threads waiting on main clock
      unblockedThreads ++= blockedThreads.getOrElse(dut.clock, Seq())
      blockedThreads.remove(dut.clock)
      clockCounter.put(dut.clock, getClockCycle(dut.clock) + 1)

      // println(s"clock step")  // TODO: toggle-able debug mode

      // TODO: allow dependent clocks to step based on test stimulus generator
      // Unblock threads waiting on dependent clock
      require((blockedThreads.keySet - dut.clock) subsetOf lastClockValue.keySet)
      val untrackClocks = lastClockValue.keySet -- blockedThreads.keySet
      for (untrackClock <- untrackClocks) {  // purge unused clocks
        lastClockValue.remove(untrackClock)
      }
      lastClockValue foreach { case (clock, lastValue) =>
        val currentValue = getClock(clock)
        if (currentValue != lastValue) {
          lastClockValue.put(clock, currentValue)
          if (currentValue) {  // rising edge
            unblockedThreads ++= blockedThreads.getOrElse(clock, Seq())
            blockedThreads.remove(clock)
            threadingChecker.advanceClock(clock)

            clockCounter.put(clock, getClockCycle(clock) + 1)
          }
        }
      }

      // Actually run things
      runThreads(unblockedThreads)

      // Propagate exceptions
      if (!interruptedException.isEmpty()) {
        throw interruptedException.poll()
      }

      threadingChecker.timestep()
      Context().env.checkpoint()
      tester.step(1)
    }

    for (thread <- allThreads.clone()) {
      // Kill the threads using an InterruptedException
      if (thread.thread.isAlive) {
        thread.thread.interrupt()
      }
    }
  }
}

object TreadleExecutive {
  import chisel3.internal.firrtl.Circuit
  import chisel3.experimental.BaseModule

  import firrtl._

  def getTopModule(circuit: Circuit): BaseModule = {
    (circuit.components find (_.name == circuit.name)).get.id
  }

  def start[T <: MultiIOModule](
    dutGen: => T,
    options: Option[ExecutionOptionsManager
            with HasChiselExecutionOptions
            with HasFirrtlOptions
            with HasInterpreterSuite
            with HasTreadleSuite] = None): BackendInstance[T] = {
    val optionsManager = options match  {
      case Some(o: ExecutionOptionsManager) => o

      case None =>
        new ExecutionOptionsManager("chisel3")
          with HasChiselExecutionOptions with HasFirrtlOptions with HasInterpreterSuite with HasTreadleSuite {
          commonOptions = CommonOptions(targetDirName = "test_run_dir")
        }
    }
    // the backend must be firrtl if we are here, therefore we want the firrtl compiler
    optionsManager.firrtlOptions = optionsManager.firrtlOptions.copy(compilerName = "low")

    chisel3.Driver.execute(optionsManager, () => dutGen) match {
      case ChiselExecutionSuccess(Some(circuit), _, Some(firrtlExecutionResult)) =>
        firrtlExecutionResult match {
          case FirrtlExecutionSuccess(_, compiledFirrtl) =>
            val dut = getTopModule(circuit).asInstanceOf[T]
            val interpretiveTester = new TreadleTester(compiledFirrtl, optionsManager)
            new TreadleBackend(dut, interpretiveTester)
          case FirrtlExecutionFailure(message) =>
            throw new Exception(s"FirrtlBackend: failed firrtl compile message: $message")
        }
      case _ =>
        throw new Exception("Problem with compilation")
    }
  }
}
