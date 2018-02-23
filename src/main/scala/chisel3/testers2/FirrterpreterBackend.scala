// See LICENSE for license details.

package chisel3.testers2

import chisel3._

import java.util.concurrent.{SynchronousQueue, TimeUnit}
import scala.collection.mutable

import firrtl_interpreter._


class FirrterpreterBackend[T <: Module](dut: T, tester: InterpretiveTester)
    extends BackendInstance[T] with ThreadedBackend {
  def getModule() = dut

  /** Returns a Seq of (data reference, fully qualified element names) for the input.
    * name is the name of data
    */
  protected def getDataNames(name: String, data: Data): Seq[(Data, String)] = Seq(data -> name) ++ (data match {
    case e: Element => Seq()
    case b: Record => b.elements.toSeq flatMap {case (n, e) => getDataNames(s"${name}_$n", e)}
    case v: Vec[_] => v.zipWithIndex flatMap {case (e, i) => getDataNames(s"${name}_$i", e)}
  })

  // TODO: the naming facility should be part of infrastructure not backend
  protected val portNames = getDataNames("io", dut.io).toMap
  protected def resolveName(signal: Data) =
    portNames.getOrElse(signal, signal.toString())

  protected val threadingChecker = new ThreadingChecker()

  override def pokeBits(signal: Bits, value: BigInt, priority: Int): Unit = {
    if (threadingChecker.doPoke(signal, priority, new Throwable)) {
      tester.poke(portNames(signal), value)
    }
  }

  override def peekBits(signal: Bits, stale: Boolean): BigInt = {
    require(!stale, "Stale peek not yet implemented")

    threadingChecker.doPeek(signal, new Throwable)
    tester.peek(portNames(signal))
  }

  override def expectBits(signal: Bits, value: BigInt, stale: Boolean): Unit = {
    require(!stale, "Stale peek not yet implemented")

    Context().env.testerExpect(value, peekBits(signal, stale), resolveName(signal), None)
  }

  protected val clockCounter = mutable.HashMap[Clock, Int]()
  protected def getClockCycle(clk: Clock): Int = {
    clockCounter.getOrElse(clk, 0)
  }

  protected val lastClockValue = mutable.HashMap[Clock, Boolean]()
  protected val pendingClocks = mutable.ArrayBuffer[Clock]()

  protected def scheduler() {
    while (activeThreads.isEmpty) {
      threadingChecker.finishTimestep()

      // If there are no pending clocks, check if any have been enabled
      if (pendingClocks.isEmpty) {
        // TODO: purge unused clocks instead of still continuing to track them
        val waitingClocks = blockedThreads.keySet ++ lastClockValue.keySet - dut.clock
        for (waitingClock <- waitingClocks) {
          val currentClockVal = tester.peek(portNames(waitingClock)).toInt match {
            case 0 => false
            case 1 => true
          }
          if (lastClockValue.getOrElseUpdate(waitingClock, currentClockVal) != currentClockVal) {
            lastClockValue.put(waitingClock, currentClockVal)
            if (currentClockVal == true) {
              pendingClocks += waitingClock
            }
          }
        }
      }

      // If so, run tasks
      if (!pendingClocks.isEmpty) {
        val activeClock = pendingClocks.head
        pendingClocks.trimStart(1)
        clockCounter.put(activeClock, getClockCycle(activeClock) + 1)
        threadingChecker.newTimestep(activeClock)

        activeThreads ++= blockedThreads.getOrElse(activeClock, Seq())
        blockedThreads.remove(activeClock)
      } else {  // Otherwise, step the main clock
        tester.step(1)
        clockCounter.put(dut.clock, getClockCycle(dut.clock) + 1)
        threadingChecker.newTimestep(dut.clock)

        activeThreads ++= blockedThreads.getOrElse(dut.clock, Seq())
        blockedThreads.remove(dut.clock)
      }
    }

    val nextThread = activeThreads.head
    currentThread = Some(nextThread)
    activeThreads.trimStart(1)
    nextThread.waiting.release()
  }

  override def step(signal: Clock, cycles: Int): Unit = {
    // TODO: clock-dependence
    // TODO: maybe a fast condition for when threading is not in use?
    for (_ <- 0 until cycles) {
      val thisThread = currentThread.get
      threadingChecker.finishThread(thisThread, signal)
      // TODO this also needs to be called on thread death

      blockedThreads.put(signal, blockedThreads.getOrElseUpdate(signal, Seq()) :+ thisThread)
      scheduler()
      thisThread.waiting.acquire()
    }
  }

  protected var scalatestThread: Option[Thread] = None
  protected val interruptedException = new SynchronousQueue[Throwable]()

  protected def onException(e: Throwable) {
    scalatestThread.get.interrupt()
    interruptedException.offer(e, 10, TimeUnit.SECONDS)
  }

  override def run(testFn: T => Unit): Unit = {
    tester.poke("reset", 1)
    tester.step(1)
    tester.poke("reset", 0)

    val mainThread = fork( {
      testFn(dut)
    }, true)

    require(activeThreads.length == 1)  // only thread should be main
    activeThreads.trimStart(1)
    currentThread = Some(mainThread)
    scalatestThread = Some(Thread.currentThread())

    mainThread.waiting.release()
    try {
      mainThread.thread.join()
    } catch {
      case e: InterruptedException =>
        throw interruptedException.poll(10, TimeUnit.SECONDS)
    }

    scalatestThread = None
    currentThread = None

    for (thread <- allThreads.clone()) {
      // Kill the threads using an InterruptedException
      if (thread.thread.isAlive) {
        thread.thread.interrupt()
      }
    }
  }
}

object Firrterpreter {
  import chisel3.internal.firrtl.Circuit
  import chisel3.experimental.BaseModule

  import firrtl._

  def getTopModule(circuit: Circuit): BaseModule = {
    (circuit.components find (_.name == circuit.name)).get.id
  }

  def start[T <: Module](dutGen: => T): BackendInstance[T] = {
    val optionsManager = new ExecutionOptionsManager("chisel3")
        with HasChiselExecutionOptions with HasFirrtlOptions with HasInterpreterSuite {
      firrtlOptions = FirrtlExecutionOptions(
        compilerName = "low"
      )
    }

    chisel3.Driver.execute(optionsManager, () => dutGen) match {
      case ChiselExecutionSuccess(Some(circuit), _, Some(firrtlExecutionResult)) =>
        firrtlExecutionResult match {
          case FirrtlExecutionSuccess(_, compiledFirrtl) =>
            val dut = getTopModule(circuit).asInstanceOf[T]
            val interpretiveTester = new InterpretiveTester(compiledFirrtl, optionsManager)
            new FirrterpreterBackend(dut, interpretiveTester)
          case FirrtlExecutionFailure(message) =>
            throw new Exception(s"FirrtlBackend: failed firrtl compile message: $message")
        }
      case _ =>
        throw new Exception("Problem with compilation")
    }
  }
}
