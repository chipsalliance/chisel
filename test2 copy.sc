//> using repository "sonatype-s01:snapshots"
//> using scala "2.13.14"
//> using dep "org.chipsalliance::chisel:latest.integration"
//> using plugin "org.chipsalliance:::chisel-plugin:latest.integration"
//> using options "-unchecked", "-deprecation", "-language:reflectiveCalls", "-feature", "-Xcheckinit", "-Ymacro-annotations"

// import chisel3._
// // _root_ disambiguate from package chisel3.util.circt if user imports chisel3.util._
// import _root_.circt.stage.ChiselStage
import scala.concurrent._
import scala.concurrent.duration._
import java.util.concurrent.{CyclicBarrier, Phaser}

case class Task(runnable: () => Unit, id: Int)

class Scheduler(
  stepBarrier: StepBarrier,
  tasks:       Seq[Task]
)(
  implicit ec: scala.concurrent.ExecutionContext) {

  def run() = {
    // first register for the barrier
    stepBarrier.register()
    val threads = tasks.map(wrapTask(_))
    try {
      while (!stepBarrier.isDone) {
        println(s"-- scheduler: waiting for all --")
        System.out.flush()
        stepBarrier.await()
        println(s"---- Scheduler: clock.step ----\n")
        System.out.flush()
        Thread.sleep(100)
        stepBarrier.completeStep()
      }
      // Wait for all to complete
      println("All steps completed.")
    } finally {
      stepBarrier.deRegister()
    }
    println("waiting for all threads to complete")
    Await.result(Future.sequence(threads), Duration.Inf)
  }

  def wrapTask(task: Task): Future[Unit] = Future {
    stepBarrier.register()
    try {
      task.runnable()
    } finally {
      println(s"Worker ${task.id}: de-registering")
      stepBarrier.deRegister()
    }
  }
}

class StepBarrier() {
  private final val preStep = new Phaser(0)
  private final val postStep = new Phaser(0)

  def register(): Unit = synchronized {
    preStep.register()
    postStep.register()
  }

  def isDone: Boolean = synchronized {
    preStep.getUnarrivedParties() <= 1 || postStep
      .getUnarrivedParties() <= 1 || preStep.isTerminated || postStep.isTerminated
  }

  def deRegister(): Unit = {
    /// or awaitAdvance(arriveAndDeregister()) ???
    preStep.awaitAdvance(preStep.arriveAndDeregister())
    postStep.awaitAdvance(postStep.arriveAndDeregister())
    // preStep.arriveAndDeregister()
    // postStep.arriveAndDeregister() /// ?????
  }

  def step(): Unit = { // synchronization ???? (should not block after first advance)
    preStep.arriveAndAwaitAdvance()
    postStep.arriveAndAwaitAdvance()
  }

  def completeStep(): Unit = synchronized {
    postStep.arriveAndAwaitAdvance()
  }

  def await(): Unit = synchronized {
    println(s"Scheduler: waiting ...")
    preStep.arriveAndAwaitAdvance()
  }
}

object LockStepExample extends App {

  val stepBarrier = new StepBarrier()

  def step(): Unit = {
    stepBarrier.step() // TODO get barrier from dynamic thread context?
  }

  // Worker function
  def worker(id: Int) = {
    for (i <- 1 to (5)) {
      println(s"  Worker $id: trying step $i")
      System.out.flush()
      step()
    }
  }

  // Start worker futures
  val tasks = (1 to 5).map(i => Task(() => worker(i), i))

  // Start the scheduler

  import scala.concurrent.ExecutionContext.Implicits.global
  val scheduler = new Scheduler(stepBarrier, tasks)

  scheduler.run()

}

LockStepExample.main(Array.empty)
