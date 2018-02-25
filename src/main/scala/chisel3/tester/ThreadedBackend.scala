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

    // Batch up pokes and peeks within a thread so they can be associated with a clock
    protected val threadPoke = mutable.HashMap[Data, (Int, Throwable)]()  // data poked to priority
    protected val threadPeek = mutable.HashMap[Data, Throwable]()

    // Within a timestep, need to enforce ordering-independence of peeks and pokes
    protected val timestepPokePriority = mutable.HashMap[Data, Int]()  // data poked to priority

    protected val timestepPokes = mutable.HashMap[Data, Seq[(Int, Clock, TesterThread, Throwable)]]()  // data poked to thread, priority, expiry clock
    protected val timestepPeeks = mutable.HashMap[Data, Seq[(Clock, TesterThread, Throwable)]]()  // data peeked to thread, expiry clock

    // Between timesteps, maintain a list of accesses that lock out changes in values
    protected val lockedData = mutable.HashMap[Data, Seq[(Clock, Throwable)]]()

    // Logs the poke for later checking, returns whether to execute it or not (based on
    // priority on the current timestep)
    def doPoke(signal: Data, priority: Int, trace: Throwable): Boolean = {
      // Only record the poke if it takes effect in the context of the current thread
      // (even if it doesn't happen because something else in the timestep stomped over it).
      // This allows cross-thread pokes to be validated in the finishTimestep() phase.
      val threadPokeOk = threadPoke.get(signal) match {
        case Some((prevPriority: Int, _)) if priority <= prevPriority => true
        case _ => true
      }
      if (threadPokeOk) {
        threadPoke.put(signal, (priority, trace))
        if (priority <= timestepPokePriority.getOrElse(signal, Int.MaxValue)) {
          timestepPokePriority.put(signal, priority)
          true
        } else {
          false
        }
      } else {
        false
      }
    }

    // Logs the peek for later checking
    def doPeek(signal: Data, trace: Throwable): Unit = {
      threadPeek.put(signal, trace)
    }

    // Marks the current thread as finished, associating a thread and clock with the peeks and pokes
    // and moves them to the timestep. No checks performed here.
    def finishThread(thread: TesterThread, clock: Clock): Unit = {
      threadPoke.foreach { case (signal, (priority, trace)) =>
        val thisEntry = (priority, clock, thread, trace)
        timestepPokes.put(signal, timestepPokes.getOrElse(signal, Seq()) :+ thisEntry)
      }
      threadPoke.clear()
      threadPeek.foreach { case (signal, trace) =>
        val thisEntry = (clock, thread, trace)
        timestepPeeks.put(signal, timestepPeeks.getOrElse(signal, Seq()) :+ thisEntry)
      }
      threadPeek.clear()
    }

    // Finishes the current timestep and checks that no conditions were violated.
    def finishTimestep(): Unit = {
      // Check sequences of peeks and pokes
      val allSignals = timestepPokes.keySet ++ timestepPeeks.keySet
      for (signal <- allSignals) {
        val pokes = timestepPokes.getOrElse(signal, Seq())
        val peeks = timestepPeeks.getOrElse(signal, Seq())

        // Check poke -> poke of same priority
        // TODO: this assumes one entry per thread, which is currently true but not enforced by the data structure
        val (singlePokes, multiPokes) = pokes.groupBy { case (priority, clock, thread, trace) => priority }
          .span { case (priority, data) => data.length == 1 }
          //.filter { case (priority, seq) => seq.length > 1 }
        // TODO: better error reporting
        if (!multiPokes.isEmpty) {
          throw new ThreadOrderDependentException(s"conflicting pokes on signal $signal: $multiPokes")
        }

        // TODO: check same clock for all pokes

        // Check any combination of pokes and peeks from different threads
        val effectivePoke = singlePokes.map { case (priority, seq) =>
          require(seq.length == 1)
          seq.head  // simplify by extracting only element of seq
        }.reduceOption[(Int, Clock, TesterThread, Throwable)] {
          case ((p1, c1, th1, tr1), (p2, c2, th2, tr2)) if p1 < p2 => (p1, c1, th1, tr1)
          case ((p1, c1, th1, tr1), (p2, c2, th2, tr2)) if p2 < p1 => (p2, c2, th2, tr2)
        }

        effectivePoke match {
          case Some((_, _, pokeThread, pokeTrace)) =>
            val peeksFromNonPokeThread = peeks.map {
              case (_, peekThread, peekTrace) if pokeThread != peekThread => Some(peekThread, peekTrace)
              case _ => None }.flatten
            if (!peeksFromNonPokeThread.isEmpty) {
              throw new ThreadOrderDependentException(s"conflicting peeks/pokes on signal $signal: poke from $effectivePoke, peeks from $peeksFromNonPokeThread")
            }
          case None =>
        }

        // Check pokes against inter-timestep lockouts
        (effectivePoke, lockedData.get(signal)) match {
          case (Some((_, _, pokeThread, pokeTrace)), Some(lockedDatas)) if !lockedDatas.isEmpty =>
            throw new SignalOverwriteException(s"poke on $signal conflicts with previous accesses: $lockedDatas")
          case _ =>
        }

        // TODO: check combinational shadows of pokes and peeks

        // Transfer to locked data for future timesteps
        effectivePoke.map { case (_, clock, _, trace) =>
          val thisEntry = (clock, trace)
          lockedData.put(signal, lockedData.getOrElse(signal, Seq()) :+ thisEntry)
        }
        peeks.map { case (clock, _, trace) =>
          val thisEntry = (clock, trace)
          lockedData.put(signal, lockedData.getOrElse(signal, Seq()) :+ thisEntry)
        }
      }
      timestepPokePriority.clear()
      timestepPokes.clear()
      timestepPeeks.clear()
    }

    // Starts a new timestep, clearing out locked signals that were due to expire.
    def newTimestep(clock: Clock): Unit = {
      for (signal <- lockedData.keysIterator) {
        lockedData.put(signal, lockedData.get(signal).get.filter{ case (lockedClock, _) => clock != lockedClock })
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
          val result = try {
            runnable
          } catch {
            case e: InterruptedException => throw e  // propagate to upper level handler
            case e: Exception => onException(e)
              waiting.acquire()
            case e: Error => onException(e)
              waiting.acquire()
          }
          threadFinished(TesterThread.this)
        } catch {
          case e: InterruptedException =>  // currently used as a signal to stop the thread
            // TODO: allow other uses for InterruptedException?
        }
      }
    })
  }

  protected var currentThread: Option[TesterThread] = None

  protected val activeThreads = mutable.ArrayBuffer[TesterThread]()  // list of threads scheduled for sequential execution
  protected val blockedThreads = mutable.HashMap[Clock, Seq[TesterThread]]()  // threads blocking on a clock edge
  protected val joinedThreads = mutable.HashMap[TesterThread, Seq[TesterThread]]()  // threads blocking on another thread
  protected val allThreads = mutable.ArrayBuffer[TesterThread]()  // list of all threads

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
    thread.done = true
    allThreads -= thread
    joinedThreads.remove(thread) match {
      case Some(testerThreads) => activeThreads ++= testerThreads
      case None =>
    }
    scheduler()
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
    val threadTyped = thread.asInstanceOf[TesterThread]  // TODO get rid of this, perhap by making it typesafe
    if (!threadTyped.done) {
      joinedThreads.put(threadTyped, joinedThreads.getOrElseUpdate(threadTyped, Seq()) :+ thisThread)
      scheduler()
      thisThread.waiting.acquire()
    }
  }
}
