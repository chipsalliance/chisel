//> using repository "sonatype-s01:snapshots"
//> using scala "2.13.14"
//> using dep "org.chipsalliance::chisel:latest.integration"
//> using plugin "org.chipsalliance:::chisel-plugin:latest.integration"
//> using options "-unchecked", "-deprecation", "-language:reflectiveCalls", "-feature", "-Xcheckinit", "-Ymacro-annotations"

import chisel3._
// _root_ disambiguate from package chisel3.util.circt if user imports chisel3.util._
import _root_.circt.stage.ChiselStage

class Foo extends Module {
  val a, b, c = IO(Input(Bool()))
  val d, e, f = IO(Input(Bool()))
  val foo, bar = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))

  val myReg = RegInit(0.U(8.W))
  out := myReg

  when(a && b && c) {
    myReg := foo
  }
  when(d && e && f) {
    myReg := bar
  }
}

// println("here")

object Main extends App {
  println(
    ChiselStage.emitSystemVerilog(
      gen = new Foo,
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
    )
  )
}

import scala.concurrent._

implicit val ec: scala.concurrent.ExecutionContext = scala.concurrent.ExecutionContext.global

case class LockStepSynchronizer() {
  var count = 0
  var stop = true
  var n: Int = 0

  def await(): Unit = {
    synchronized {
      count += 1
      notifyAll()
      while (count < n) {
        wait()
      }
      stop = true
    }
    synchronized {
      while (stop) {
        wait()
      }
    }
  }

  def awaitMaster(): Unit = {
    synchronized {
      while (count < n) {
        wait()
      }
    }
  }

  def decrementN(): Unit = {
    synchronized {
      if (n > 0) {
        n -= 1
        notifyAll()
      }
    }
  }
  def setN(nn: Int): Unit = {
    synchronized {
      n = nn
      notifyAll()
    }
  }
  def reset(): Unit = {
    synchronized {
      count = 0
      stop = false
      notifyAll()
    }
  }
}

sealed private case class FF(runnables: (() => Unit)*) {
  def activate: Future[Unit] = Future {
    runnables.foreach(r => r())
  }
}

sealed class ForkBuilder(barrier: LockStepSynchronizer, tasks: Seq[FF]) {
  def this(barrier: LockStepSynchronizer, runnable: () => Unit) = this(barrier, Seq(FF(runnable, () => barrier.decrementN())))

  def fork(runnable: => Unit): ForkBuilder =
    new ForkBuilder(barrier, tasks :+ FF(() => runnable, () => barrier.decrementN()))

  def join(): Unit = {
    require(tasks.length > 1, "At least 2 fork blocks are needed for join")

    assert(barrier.count == 0, "LockStepSynchronizer count is not zero")
    barrier.setN(tasks.length)
    val threads = tasks.map(_.activate)

    def runScheduler(): Unit = {
      while (true) {
        barrier.awaitMaster()
        if (threads.forall(_.isCompleted))
          return
        println(s"----Scheduler stepped----")
        System.out.flush()
        Thread.sleep(100)
        barrier.reset()
      }
    }
    val combined = Future.sequence(threads :+ Future { runScheduler() }) //.map(_ => ())

    Await.ready(combined, scala.concurrent.duration.Duration.Inf)
  }
}

object fork {
  val barrier = new LockStepSynchronizer()

  def apply(runnable: => Unit): ForkBuilder =
    new ForkBuilder(barrier, () => runnable)
}

fork {
  for (i <- 0 until 10) {
    println(s"Task1 stepping $i")
    System.out.flush()
    step()
    println(s"Task1 stepping again $i")
    System.out.flush()
    step()
  }
}.fork {
  for (i <- 0 until 13) {
    println(s"Task2 stepping $i")
    System.out.flush()
    step()
  }
}.join()

def step() = {
  fork.barrier.await()
}
