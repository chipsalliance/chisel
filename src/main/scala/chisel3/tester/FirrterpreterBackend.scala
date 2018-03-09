// See LICENSE for license details.

package chisel3.tester

import chisel3._
import chisel3.tester.TesterUtils.getPortNames
import java.util.concurrent.{Semaphore, SynchronousQueue, TimeUnit}
import scala.collection.mutable

import firrtl_interpreter._

class FirrterpreterBackend[T <: Module](dut: T, tester: InterpretiveTester)
    extends BackendInstance[T] with ThreadedBackend {
  def getModule() = dut

  // TODO: the naming facility should be part of infrastructure not backend
  protected val portNames = getPortNames(dut)
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

  protected def scheduler() {
    var testDone: Boolean = false  // set at the end of the clock cycle that the main thread dies on

    while (activeThreads.isEmpty && !testDone) {
      threadingChecker.finishTimestep()
      Context().env.checkpoint()
      tester.step(1)
      clockCounter.put(dut.clock, getClockCycle(dut.clock) + 1)

      if (mainTesterThread.get.done) {
        testDone = true
      }

      threadingChecker.newTimestep(dut.clock)

      // Unblock threads waiting on main clock
      activeThreads ++= blockedThreads.getOrElse(dut.clock, Seq())
      blockedThreads.remove(dut.clock)

      // Unblock threads waiting on dependent clocks
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
            activeThreads ++= blockedThreads.getOrElse(waitingClock, Seq())
            blockedThreads.remove(waitingClock)
            threadingChecker.newTimestep(waitingClock)

            clockCounter.put(waitingClock, getClockCycle(waitingClock) + 1)
          }
        }
      }
    }

    if (!testDone) {  // if test isn't over, run next thread
      val nextThread = activeThreads.head
      currentThread = Some(nextThread)
      activeThreads.trimStart(1)
      nextThread.waiting.release()
    } else {  // if test is done, return to the main scalatest thread
      scalatestWaiting.release()
    }

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
  protected var mainTesterThread: Option[TesterThread] = None
  protected val scalatestWaiting = new Semaphore(0)
  protected val interruptedException = new SynchronousQueue[Throwable]()

  protected def onException(e: Throwable) {
    scalatestThread.get.interrupt()
    interruptedException.offer(e, 10, TimeUnit.SECONDS)
  }

  override def run(testFn: T => Unit): Unit = {
    tester.poke("reset", 1)
    tester.step(1)
    tester.poke("reset", 0)

    val mainThread = fork(
      testFn(dut)
    )

    require(activeThreads.length == 1)  // only thread should be main
    activeThreads.trimStart(1)
    currentThread = Some(mainThread)
    scalatestThread = Some(Thread.currentThread())
    mainTesterThread = Some(mainThread)

    mainThread.waiting.release()
    try {
      scalatestWaiting.acquire()
    } catch {
      case e: InterruptedException =>
        throw interruptedException.poll(10, TimeUnit.SECONDS)
    }

    mainTesterThread = None
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

  def start[T <: Module](dutGen: => T, options: Option[TesterOptionsManager] = None): BackendInstance[T] = {
    val optionsManager = options match  {
      case Some(o: ExecutionOptionsManager) => o

      case None =>
        new ExecutionOptionsManager("chisel3")
          with HasChiselExecutionOptions with HasFirrtlOptions with HasInterpreterSuite {
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
