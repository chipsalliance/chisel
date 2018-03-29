// See LICENSE for license details.

package chisel3.tester

import java.util.concurrent.{Semaphore, SynchronousQueue, TimeUnit}
import scala.collection.mutable

import chisel3._

/** Common utility functions for backends implementing concurrency by threading.
  * The backend must invoke concurrency functions as appropriate, eg during step() calls
  */
trait ThreadedBackend {
  protected class ThreadingChecker {
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
    }
    case class SignalPokeRecord(timescope: Timescope, priority: Int, value: BigInt,
        trace: Throwable) extends PokeRecord
    case class XPokeRecord() extends PokeRecord {
      def priority: Int = Int.MaxValue
    }
    case class PeekRecord(thread: TesterThread, trace: Throwable)

    protected val threadTimescopes = mutable.HashMap[TesterThread, mutable.ListBuffer[Timescope]]()

    // Active pokes on a signal, map of wire -> priority -> timescope
    // The stack of timescopes must all be from the same thread, this invariant must be checked before
    // pokes are committed to this data structure.
    protected val signalPokes = mutable.HashMap[Data, mutable.HashMap[Int, mutable.ListBuffer[Timescope]]]()

    // Between timesteps, maintain a list of accesses that lock out changes in values
    protected val lockedData = mutable.HashMap[Data, Seq[(Clock, Throwable)]]()

    // All poke operations happening on this timestep, including end-of-timescope reverts.
    protected val timestepPokes = mutable.HashMap[Data, mutable.ListBuffer[PokeRecord]]()
    protected val timestepPeeks = mutable.HashMap[Data, mutable.ListBuffer[PeekRecord]]()

    /**
     * Logs a poke operation for later checking.
     * Returns whether to execute it, based on priorities compared to other active pokes.
     */
    def doPoke(thread: TesterThread, signal: Data, value: BigInt, priority: Int, trace: Throwable): Boolean = {
      val timescope = threadTimescopes(thread).last
      val pokeRecord = SignalPokeRecord(timescope, priority, value, trace)
      timestepPokes.getOrElseUpdate(signal, mutable.ListBuffer[PokeRecord]()).append(
          pokeRecord)
      timescope.pokes.put(signal, pokeRecord)
      signalPokes.getOrElseUpdate(signal, mutable.HashMap[Int, mutable.ListBuffer[Timescope]]())
          .getOrElseUpdate(priority, mutable.ListBuffer[Timescope]())
          .append(timescope)
      priority <= (signalPokes(signal).keys foldLeft Int.MaxValue)(Math.min)
    }

    /**
     * Logs a peek operation for later checking.
     */
    def doPeek(thread: TesterThread, signal: Data, trace: Throwable): Unit = {
      timestepPeeks.getOrElseUpdate(signal, mutable.ListBuffer[PeekRecord]()).append(
          PeekRecord(thread, trace))
    }

    /**
     * Creates a new timescope in the specified thread.
     */
    def newTimescope(parent: TesterThread): Timescope = {
      val newTimescope = new Timescope(parent)
      threadTimescopes.getOrElseUpdate(parent, mutable.ListBuffer[Timescope]()).append(
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
          case None => XPokeRecord()
        })
      }

      // Register those pokes as happening on this timestep
      revertMap foreach { case (data, pokeRecord) =>
        timestepPokes.getOrElseUpdate(data, mutable.ListBuffer[PokeRecord]()).append(
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

    }

    /**
     * Starts a new timestep, checking if there were any conflicts on the previous timestep (and
     * throwing exceptions if there were).
     */
    def timestep(): Unit = {
      // commit the last poke and peek on this timestep from each thread

      // check overlapped pokes from different threads with same priority

      // check poke | peek dependencies

    }


//    // Logs the poke for later checking, returns whether to execute it or not (based on
//    // priority on the current timestep)
//    def doPoke(signal: Data, priority: Int, trace: Throwable): Boolean = {
//      // Only record the poke if it takes effect in the context of the current thread
//      // (even if it doesn't happen because something else in the timestep stomped over it).
//      // This allows cross-thread pokes to be validated in the finishTimestep() phase.
//      val threadPokeOk = threadPoke.get(signal) match {
//        case Some((prevPriority: Int, _)) if priority <= prevPriority => true
//        case _ => true
//      }
//      if (threadPokeOk) {
//        threadPoke.put(signal, (priority, trace))
//        if (priority <= timestepPokePriority.getOrElse(signal, Int.MaxValue)) {
//          timestepPokePriority.put(signal, priority)
//          true
//        } else {
//          false
//        }
//      } else {
//        false
//      }
//    }
//
//    // Logs the peek for later checking
//    def doPeek(signal: Data, trace: Throwable): Unit = {
//      threadPeek.put(signal, trace)
//    }
//
//    // Marks the current thread as finished, associating a thread and clock with the peeks and pokes
//    // and moves them to the timestep. No checks performed here.
//    def finishThread(thread: TesterThread, clock: Clock): Unit = {
//      threadPoke.foreach { case (signal, (priority, trace)) =>
//        val thisEntry = (priority, clock, thread, trace)
//        timestepPokes.put(signal, timestepPokes.getOrElse(signal, Seq()) :+ thisEntry)
//      }
//      threadPoke.clear()
//      threadPeek.foreach { case (signal, trace) =>
//        val thisEntry = (clock, thread, trace)
//        timestepPeeks.put(signal, timestepPeeks.getOrElse(signal, Seq()) :+ thisEntry)
//      }
//      threadPeek.clear()
//    }
//
//    // Finishes the current timestep and checks that no conditions were violated.
//    def finishTimestep(): Unit = {
//      // Check sequences of peeks and pokes
//      val allSignals = timestepPokes.keySet ++ timestepPeeks.keySet
//      for (signal <- allSignals) {
//        val pokes = timestepPokes.getOrElse(signal, Seq())
//        val peeks = timestepPeeks.getOrElse(signal, Seq())
//
//        // Check poke -> poke of same priority
//        // TODO: this assumes one entry per thread, which is currently true but not enforced by the data structure
//        val (singlePokes, multiPokes) = pokes.groupBy { case (priority, clock, thread, trace) => priority }
//          .span { case (priority, data) => data.length == 1 }
//          //.filter { case (priority, seq) => seq.length > 1 }
//        // TODO: better error reporting
//        if (!multiPokes.isEmpty) {
//          throw new ThreadOrderDependentException(s"conflicting pokes on signal $signal: $multiPokes")
//        }
//
//        // TODO: check same clock for all pokes
//
//        // Check any combination of pokes and peeks from different threads
//        val effectivePoke = singlePokes.map { case (priority, seq) =>
//          require(seq.length == 1)
//          seq.head  // simplify by extracting only element of seq
//        }.reduceOption[(Int, Clock, TesterThread, Throwable)] {
//          case ((p1, c1, th1, tr1), (p2, c2, th2, tr2)) if p1 < p2 => (p1, c1, th1, tr1)
//          case ((p1, c1, th1, tr1), (p2, c2, th2, tr2)) if p2 < p1 => (p2, c2, th2, tr2)
//        }
//
//        effectivePoke match {
//          case Some((_, _, pokeThread, pokeTrace)) =>
//            val peeksFromNonPokeThread = peeks.map {
//              case (_, peekThread, peekTrace) if pokeThread != peekThread => Some(peekThread, peekTrace)
//              case _ => None }.flatten
//            if (!peeksFromNonPokeThread.isEmpty) {
//              throw new ThreadOrderDependentException(s"conflicting peeks/pokes on signal $signal: poke from $effectivePoke, peeks from $peeksFromNonPokeThread")
//            }
//          case None =>
//        }
//
//        // Check pokes against inter-timestep lockouts
//        (effectivePoke, lockedData.get(signal)) match {
//          case (Some((_, _, pokeThread, pokeTrace)), Some(lockedDatas)) if !lockedDatas.isEmpty =>
//            throw new SignalOverwriteException(s"poke on $signal conflicts with previous accesses: $lockedDatas")
//          case _ =>
//        }
//
//        // TODO: check combinational shadows of pokes and peeks
//
//        // Transfer to locked data for future timesteps
//        effectivePoke.map { case (_, clock, _, trace) =>
//          val thisEntry = (clock, trace)
//          lockedData.put(signal, lockedData.getOrElse(signal, Seq()) :+ thisEntry)
//        }
//        peeks.map { case (clock, _, trace) =>
//          val thisEntry = (clock, trace)
//          lockedData.put(signal, lockedData.getOrElse(signal, Seq()) :+ thisEntry)
//        }
//      }
//      timestepPokePriority.clear()
//      timestepPokes.clear()
//      timestepPeeks.clear()
//    }
//
//    // Starts a new timestep, clearing out locked signals that were due to expire.
//    def newTimestep(clock: Clock): Unit = {
//      for (signal <- lockedData.keysIterator) {
//        lockedData.put(signal, lockedData.get(signal).get.filter{ case (lockedClock, _) => clock != lockedClock })
//      }
//    }
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
