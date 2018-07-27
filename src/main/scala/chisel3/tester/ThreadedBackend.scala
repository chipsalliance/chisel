// See LICENSE for license details.

package chisel3.tester

import java.util.concurrent.{Semaphore, SynchronousQueue, TimeUnit}
import scala.collection.mutable

import chisel3._

/** Common utility functions for backends implementing concurrency by threading.
  * The backend must invoke concurrency functions as appropriate, eg during step() calls
  */
trait ThreadedBackend {
  // combinationalPaths: map of sink Data to all source Data nodes.
  protected class ThreadingChecker(combinationalPaths: Map[Data, Set[Data]]) {
    /** Desired threading checking behavior:
      * -> indicates following, from different thread
      * poke -> poke (higher priority): OK (result is order-independent)
      * poke -> poke (same priority): not OK (result is order-dependent)
      * poke -> peek (and reversed - equivalent to be order-independent): not OK (peek may be stale)
      * poke -> peek of something in combinational shadow (and reversed): not OK
      *
      * fringe cases: poke -> poke (higher priority), peek: probably should be OK, but unlikely to be necessary?
      *
      * Pokes and peeks are enforced through the end of the clock cycle.
      *
      * In each discrete timestep, all actions should execute and batch up errors, so that thread
      * ordering errors can be generated in addition to any (false-positive) correctness errors:
      * Start timestep by moving blocked threads to active
      *   Clear active peeks / pokes belonging to the last cycle of that clock
      * Run all threads:
      *   If other joined threads are enabled, add them to the end of the active threads list
      *   Exceptions propagate immediately
      *   Test failures are batched to the end of the timestep
      *   Peeks and pokes are recorded in a table and associated with the clock / timestep and thread
      * At the end of the timestep:
      *   Check for peek and poke ordering dependence faults
      *   Report batched test failures
      */

    protected class Timescope(val parent: TesterThread) {
      // Latest poke on a signal in this timescope
      val pokes = mutable.HashMap[Data, SignalPokeRecord]()
    }

    abstract class PokeRecord {
      def priority: Int
      def thread: TesterThread
    }
    case class SignalPokeRecord(timescope: Timescope, priority: Int, value: BigInt,
        trace: Throwable) extends PokeRecord {
      override def thread = timescope.parent
    }
    case class XPokeRecord(thread: TesterThread) extends PokeRecord {
      def priority: Int = Int.MaxValue
    }
    case class PeekRecord(thread: TesterThread, trace: Throwable)

    protected val threadTimescopes = mutable.HashMap[TesterThread, mutable.ListBuffer[Timescope]]()

    // Active pokes on a signal, map of wire -> priority -> timescope
    // The stack of timescopes must all be from the same thread, this invariant must be checked before
    // pokes are committed to this data structure.
    protected val signalPokes = mutable.HashMap[Data, mutable.HashMap[Int, mutable.ListBuffer[Timescope]]]()
    protected val revertPokes = mutable.HashMap[Data, mutable.ListBuffer[PokeRecord]]()

    // Active peeks on a signal, lasts until the specified clock advances
    protected val signalPeeks = mutable.HashMap[Clock, mutable.HashMap[Data, mutable.ListBuffer[PeekRecord]]]()

    // All poke revert operations

    /**
     * Logs a poke operation for later checking.
     * Returns whether to execute it, based on priorities compared to other active pokes.
     */
    def doPoke(thread: TesterThread, signal: Data, value: BigInt, priority: Int, trace: Throwable): Boolean = {
      val timescope = threadTimescopes(thread).last
      val pokeRecord = SignalPokeRecord(timescope, priority, value, trace)
      timescope.pokes.put(signal, pokeRecord)
      signalPokes.getOrElseUpdate(signal, mutable.HashMap())
          .getOrElseUpdate(priority, mutable.ListBuffer())
          .append(timescope)
      priority <= (signalPokes(signal).keys foldLeft Int.MaxValue)(Math.min)
    }

    /**
     * Logs a peek operation for later checking.
     */
    def doPeek(thread: TesterThread, signal: Data, clock: Clock, trace: Throwable): Unit = {
      signalPeeks.getOrElseUpdate(clock, mutable.HashMap())
          .getOrElseUpdate(signal, mutable.ListBuffer())
          .append(PeekRecord(thread, trace))
    }

    /**
     * Creates a new timescope in the specified thread.
     */
    def newTimescope(parent: TesterThread): Timescope = {
      val newTimescope = new Timescope(parent)
      threadTimescopes.getOrElseUpdate(parent, mutable.ListBuffer()).append(
          newTimescope)
      newTimescope
    }

    /**
     * Closes the specified timescope (which must be at the top of the timescopes stack in its
     * parent thread), returns a map of wires to values of any signals that need to be updated.
     */
    def closeTimescope(timescope: Timescope): Map[Data, Option[BigInt]] = {
      val timescopeList = threadTimescopes(timescope.parent)
      require(timescopeList.last == timescope)
      timescopeList.trimEnd(1)
      if (timescopeList.isEmpty) {
        threadTimescopes.remove(timescope.parent)
      }

      // Clear the timescope from signal pokes
      timescope.pokes foreach { case (data, pokeRecord) =>
        // TODO: can this be made a constant time operation?
        signalPokes(data)(pokeRecord.priority) -= timescope
        if (signalPokes(data)(pokeRecord.priority).isEmpty) {
          signalPokes(data).remove(pokeRecord.priority)
        }
        if (signalPokes(data).isEmpty) {
          signalPokes.remove(data)
        }
      }

      // Get the PeekRecords of the value to revert to
      val revertMap = timescope.pokes.toMap map { case (data, pokeRecord) =>
        (data, signalPokes.get(data) match {
          case Some(pokesMap) => pokesMap(pokesMap.keys.min).last.pokes(data)
          case None => XPokeRecord(timescope.parent)
        })
      }

      // Register those pokes as happening on this timestep
      revertMap foreach { case (data, pokeRecord) =>
        revertPokes.getOrElseUpdate(data, mutable.ListBuffer()).append(
            pokeRecord)
      }

      revertMap map { case (data, pokeRecord) => (data, pokeRecord match {
        case signal: SignalPokeRecord => Some(signal.value)
        case _: XPokeRecord => None
      } ) }
    }

    /**
     * Called upon advancing the specified clock, allowing peeks to clear
     */
    def advanceClock(clock: Clock): Unit = {
      signalPeeks.remove(clock)
    }

    /**
     * Starts a new timestep, checking if there were any conflicts on the previous timestep (and
     * throwing exceptions if there were).
     */
    def timestep(): Unit = {
      // check overlapped pokes from different threads with same priority
      signalPokes foreach { case (signal, priorityToTimescopes) =>
        priorityToTimescopes foreach { case (priority, timescopes) =>
          val threads = timescopes.map(_.parent).distinct
          if (threads.length > 1) {
            throw new ThreadOrderDependentException(s"conflicting pokes on signal $signal at priority $priority from threads $threads")
          }
        }
      }

      // check poke | peek dependencies
      // Order keys by signal instead of clock
      val signalPeeksBySignal = signalPeeks.toSeq.map({ case (clock, signalToPeeks) =>
        signalToPeeks.toSeq  // to Seq[(Data, ListBuffer[PeekRecord])]
      }).flatten.groupBy({ case (signal, peeks) =>
        signal  // group top-level Map by signal
      }).map({ case (signal, signalToPeeks) =>
        (signal, signalToPeeks.map(_._2).flatten)  // flatten Seq[(..., ListBuffer[PeekRecord])] to Seq[PeekRecord]
      })

      signalPeeksBySignal foreach { case (signal, peeks) =>
        val peekThreads = peeks.map(_.thread).toSet
        val pokeThreads = signalPokes.get(signal) match {
          case Some(priorityToTimescopes) =>
            priorityToTimescopes(priorityToTimescopes.keySet.min).map(_.parent).toSet
          case None => Set[TesterThread]()
        }
        val revertThreads = revertPokes.get(signal) match {
          case Some(pokes) => pokes.map(_.thread).toSet
          case None => Set[TesterThread]()
        }
        // TODO: better error reporting
        if (!(pokeThreads subsetOf peekThreads)) {
          throw new ThreadOrderDependentException(s"poke on $signal conflicts with previous peeks")
        }
        if (!(revertThreads subsetOf peekThreads)) {
          throw new ThreadOrderDependentException(s"revert on $signal conflicts with previous peeks")
        }
      }
    }
  }


  protected class TesterThread(runnable: => Unit) extends AbstractTesterThread {
    val waiting = new Semaphore(0)
    var done: Boolean = false

    val thread = new Thread(new Runnable {
      def run() {
        try {
          waiting.acquire()
          try {
            timescope {
              runnable
            }
          } catch {
            case e: InterruptedException => throw e  // propagate to upper level handler
            case e @ (_: Exception | _: Error) =>
              onException(e)
          }
          done = true
          threadFinished(TesterThread.this)
          scheduler()
        } catch {
          case e: InterruptedException =>  // currently used as a signal to stop the thread
            // TODO: allow other uses for InterruptedException?
        }
      }
    })
  }

  protected var currentThread: Option[TesterThread] = None
  protected val driverSemaphore = new Semaphore(0)  // blocks runThreads() while it's running

  // TODO: replace with concurrent data structures?
  protected val activeThreads = mutable.ArrayBuffer[TesterThread]()  // list of threads scheduled for sequential execution
  protected val blockedThreads = mutable.HashMap[Clock, Seq[TesterThread]]()  // threads blocking on a clock edge
  protected val joinedThreads = mutable.HashMap[TesterThread, Seq[TesterThread]]()  // threads blocking on another thread
  protected val allThreads = mutable.ArrayBuffer[TesterThread]()  // list of all threads

  /**
   * Runs the specified threads, blocking this thread while those are running.
   * Newly formed threads or unblocked join threads will also run.
   *
   * Prior to this call: caller should remove those threads from the blockedThread list.
   * TODO: does this interface suck?
   *
   * Updates internal thread queue data structures. Exceptions will also be queued through onException() calls.
   * TODO: can (should?) this provide a more functional interface? eg returning what threads are blocking on?
   */
  protected def runThreads(threads: Seq[TesterThread]) {
    activeThreads ++= threads
    scheduler()
    driverSemaphore.acquire()
  }

  /**
   * Invokes the thread scheduler, which should be done anytime a thread needs to pass time.
   * Prior to this call: caller should add itself to the blocked / joined threads list
   * (unless terminating).
   * After this call: caller should block on its semaphore (unless terminating). currentThread
   * will no longer be valid.
   *
   * Unblocks the next thread to be run, possibly also also stepping time via advanceTime().
   * When there are no more active threads, unblocks the driver thread via driverSemaphore.
   */
  protected def scheduler() {
    if (!activeThreads.isEmpty) {
      val nextThread = activeThreads.head
      currentThread = Some(nextThread)
      activeThreads.trimStart(1)
      nextThread.waiting.release()
    } else {
      currentThread = None
      driverSemaphore.release()
    }
  }

  /**
   * Called when an exception happens inside a thread.
   * Can be used to propagate the exception back up to the main thread.
   * No guarantees are made about the state of the system on an exception.
   *
   * The thread then terminates, and the thread scheduler is invoked to unblock the next thread.
   * The implementation should only record the exception, which is properly handled later.
   */
  protected def onException(e: Throwable)

  /**
   * Called on thread completion to remove this thread from the running list.
   * Does not terminate the thread, does not schedule the next thread.
   */
  protected def threadFinished(thread: TesterThread) {
    allThreads -= thread
    joinedThreads.remove(thread) match {
      case Some(testerThreads) => activeThreads ++= testerThreads
      case None =>
    }
  }

  def fork(runnable: => Unit): TesterThread = {
    val newThread = new TesterThread(runnable)
    allThreads += newThread
    activeThreads += newThread
    newThread.thread.start()
    newThread
  }

  def join(thread: AbstractTesterThread) = {
    val thisThread = currentThread.get
    val threadTyped = thread.asInstanceOf[TesterThread]  // TODO get rid of this, perhaps by making it typesafe
    if (!threadTyped.done) {
      joinedThreads.put(threadTyped, joinedThreads.getOrElseUpdate(threadTyped, Seq()) :+ thisThread)
      scheduler()
      thisThread.waiting.acquire()
    }
  }
}
