// See LICENSE for license details.

package chisel3.testers2

import chisel3._

import scala.collection.mutable.HashMap

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

  protected val clockCounter = Map[Clock, Int]()
  protected def getClockCycle(clk: Clock): Int = {
    clockCounter.getOrElse(clk, 0)
  }

  protected def scheduler() {
    threadingChecker.finishThread(currentThread.get, dut.clock)  // TODO multiclock

    if (waitingThreads.isEmpty) {
      threadingChecker.finishTimestep()
      tester.step(1)
      waitingThreads ++= threadOrder
      threadingChecker.newTimestep(dut.clock)  // TODO allow use on multiclock designs
    }

    val nextThread = waitingThreads.head
    currentThread = Some(nextThread)
    waitingThreads.trimStart(1)
    nextThread.waiting.release()
  }

  override def step(signal: Clock, cycles: Int): Unit = {
    // TODO: clock-dependence
    // TODO: maybe a fast condition for when threading is not in use?
    for (_ <- 0 until cycles) {
      val thisThread = currentThread.get
      scheduler()
      thisThread.waiting.acquire()
    }
  }

  var scalatestThread: Option[Thread] = None
  var interruptedException: Option[Throwable] = None

  protected def onException(e: Throwable) {
    interruptedException = Some(e)
    scalatestThread.get.interrupt()
  }

  override def run(testFn: T => Unit): Unit = {
    // TODO: don't hardcode in assumption of singleclock singlereset circuit
    tester.poke("reset", 1)
    tester.step(1)
    tester.poke("reset", 0)

    val mainThread = fork( {
      testFn(dut)
    }, true)

    require(waitingThreads.length == 1)  // only thread should be main
    waitingThreads.trimStart(1)
    currentThread = Some(mainThread)
    scalatestThread = Some(Thread.currentThread())

    mainThread.waiting.release()
    try {
      mainThread.thread.join()
    } catch {
      case e: InterruptedException => Thread.sleep(500)  // TODO fix synchronization issue
        throw interruptedException.get
    }

    scalatestThread = None
    currentThread = None
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
