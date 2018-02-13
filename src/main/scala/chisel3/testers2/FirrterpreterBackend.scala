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

  protected val clockCounter = Map[Clock, Int]()
  protected def getClockCycle(clk: Clock): Int = {
    clockCounter.getOrElse(clk, 0)
  }

  def resolveName(signal: Data) =
    portNames.getOrElse(signal, signal.toString())

  /** Desired threading checking behavior:
    * -> indicates following, from different thread
    * poke -> poke (higher priority): OK (result is order-independent)
    * poke -> poke (same priority): not OK (result is order-dependent)
    * poke -> peek (and reversed - equivalent to be order-independent): not OK (peek may be stale)
    * poke -> peek of something in combinational shadow (and reversed): not OK
    *
    * should also maintain optional source trace information to help debug order-dependence failures
    *
    * fringe cases: poke -> poke (higher priority), peek: probably should be OK, but unlikely to be necessary?
    */

  // List of active pokes and associated priority and poking TesterThread
  protected val activePokes = HashMap[Data, (Int, TesterThread)]()

  def poke(signal: Data, value: BigInt, priority: Int): Unit = {
    signal match {
      case signal: Bits =>
        val doPoke = activePokes.get(signal) match {
          case Some((activePriority, thread)) =>
            // Ensure final peek isn't affected by thread execution order
            if (activePriority == priority) {
              require(thread == currentThread.get)
            }
            (activePriority >= priority)
          case None => true
        }
        if (doPoke) {
          tester.poke(portNames(signal), value)
          activePokes +=((signal, (priority, currentThread.get)))
        }
    }
  }

  def peek(signal: Data): BigInt = {
    signal match {
      case signal: Bits =>
        tester.peek(portNames(signal))
    }
  }

  def stalePeek(signal: Data): BigInt = {
    throw new Exception("Stale peek not implemented yet")
  }

  def expect(signal: Data, value: BigInt): Unit = {
    Context().env.testerExpect(value, peek(signal), resolveName(signal), None)
  }

  def staleExpect(signal: Data, value: BigInt): Unit = {
    throw new Exception("Stale check not implemented yet")
  }

  protected def scheduler() {
    if (waitingThreads.isEmpty) {
      tester.step(1)
      activePokes.clear()
      waitingThreads ++= threadOrder
    }
    val nextThread = waitingThreads.head
    currentThread = Some(nextThread)
    waitingThreads.trimStart(1)
    nextThread.waiting.release()
  }

  def step(signal: Clock, cycles: Int): Unit = {
    // TODO: clock-dependence
    // TODO: maybe a fast condition for when threading is not in use?
    for (_ <- 0 until cycles) {
      val thisThread = currentThread.get
      scheduler()
      thisThread.waiting.acquire()
    }
  }

  var scalatestThread: Option[Thread] = None
  var interruptedException: Option[Exception] = None

  protected def onException(e: Exception) {
    interruptedException = Some(e)
    scalatestThread.get.interrupt()
  }

  def run(testFn: T => Unit): Unit = {
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
      case e: InterruptedException => throw interruptedException.get
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
