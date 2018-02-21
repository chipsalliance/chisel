// See LICENSE for license details.

package chisel3.testers2

import scala.collection.mutable
import java.util.concurrent.Semaphore

import chisel3._

trait AbstractTesterThread

/** Common interface definition for tester backends
  */
trait BackendInterface {
  /** Writes a value to a writable wire.
    * Throws an exception if write is not writable.
    */
  def pokeBits(signal: Bits, value: BigInt, priority: Int): Unit

  /** Returns the current value on a wire.
    * If stale is true, returns the current combinational value (after previous pokes have taken effect).
    * If stale is false, returns the value at the beginning of the current cycle.
    */
  def peekBits(signal: Bits, stale: Boolean): BigInt

  def expectBits(signal: Bits, value: BigInt, stale: Boolean): Unit

  /** Advances the target clock by one cycle.
    */
  def step(signal: Clock, cycles: Int): Unit

  def fork(runnable: => Unit): AbstractTesterThread

  def join(thread: AbstractTesterThread)
}

/** Backend associated with a particular circuit, and can run tests
  */
trait BackendInstance[T <: Module] extends BackendInterface {
  /** Runs of tests are wrapped in this, for any special setup/teardown that needs to happen.
    * Takes the test function, which takes the module used as the testing interface.
    * TesterContext setup is done externally.
    *
    * Internal API
    */
  def run(testFn: T => Unit): Unit
}

/** Common utility functions for backends implementing concurrency by threading.
  * The backend must invoke concurrency functions as appropriate, eg during step() calls
  */
trait ThreadedBackend {
  protected class TesterThread(runnable: => Unit, isMainThread: Boolean) extends AbstractTesterThread {
    val waiting = new Semaphore(0)
    val thread = new Thread(new Runnable {
      def run() {
        waiting.acquire()
        val result = try {
          runnable
        } catch {
          case e: Exception => onException(e)
          case e: Error => onException(e)
          waiting.acquire()
        }
        if (!isMainThread) {
          threadFinished(TesterThread.this)
        } // otherwise main thread falls off the edge without running the next thread
        // TODO: should main thread at least finish its cycle?
      }
    })
  }
  var currentThread: Option[TesterThread] = None
  val waitingThreads = mutable.ArrayBuffer[TesterThread]()
  val threadOrder = mutable.ArrayBuffer[TesterThread]()

  /** Invokes the thread scheduler, which unblocks the next thread to be run
    * (and may also step simulator time).
    */
  protected def scheduler()

  /** Called when an exception happens inside a thread.
    * Can be used to propagate the exception back up to the main thread.
    * No guarantees are made about the state of the system on an exception.
    */
  protected def onException(e: Throwable)

  protected def threadFinished(thread: TesterThread) {
    threadOrder -= thread
    scheduler()
    // TODO: join notification
  }

  def fork(runnable: => Unit, isMainThread: Boolean): TesterThread = {
    val newThread = new TesterThread(runnable, isMainThread)
    threadOrder += newThread
    waitingThreads += newThread
    newThread.thread.start()
    newThread
  }

  def fork(runnable: => Unit): TesterThread = fork(runnable, false)

  def join(thread: AbstractTesterThread) = ???
}


/** Interface into the testing environment, like ScalaTest
  */
trait TestEnvInterface {
  // TODO: should these return boolean? or just assert out?
  /** Runs a test, given a specific instantiated backend.
    */
  def test[T <: Module](tester: BackendInstance[T])(testFn: T => Unit): Unit
  /** Runs a test, instantiating the default backend.
    */
  def test[T <: Module](dutGen: => T)(testFn: T => Unit): Unit = {
    test(Context.createDefaultTester(dutGen))(testFn)
  }

  /** Fails the test now.
    */
  def testerFail(msg: String): Unit
  /** Expect a specific value on a wire, calling testerFail if the expectation isn't met
    */
  def testerExpect(expected: Any, actual: Any, signal: String, msg: Option[String]): Unit
}
