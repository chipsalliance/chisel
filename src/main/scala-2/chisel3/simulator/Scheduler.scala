// SPDX-License-Identifier: Apache-2.0

package chisel3
package simulator

import scala.concurrent._

import java.util.concurrent.ForkJoinPool
import java.util.concurrent.Executor
import java.util.concurrent.TimeUnit
import java.util.concurrent.ExecutionException

private[simulator] final case class Task(runnable: () => Unit, id: Int)

private[simulator] final class FutureFactory(executor: Executor, stepBarrier: StepBarrier) {
  private val execContext = ExecutionContext.fromExecutor(executor)

  def runTask(task: Task): Future[Unit] = Future {
    try {
      task.runnable()
    } finally {
      stepBarrier.deRegister()
    }
  }(execContext)

  def runTasks(tasks: Seq[Task]): Future[IterableOnce[Unit]] = {
    val workerFutures = tasks.map(runTask)
    implicit val ec = execContext
    Future.sequence(workerFutures)
  }
}

private[simulator] final class Scheduler(val stepClock: () => Unit, enableDebugPrint: Boolean = false) {
  def debug(x: => Any): Unit = {
    if (enableDebugPrint) println(s"[Thr:${Thread.currentThread().threadId()}] $x")
  }

  def run(tasks: Seq[Task]) = {
    val stepBarrier = new StepBarrier()
    StepBarrier.withValue(stepBarrier) {
      // Start a fresh thread pool for each run to avoid DynamicVariable and other ThreadLocal issues
      // The overhead should be negligible compared to simulation times
      val executor = new ForkJoinPool()
      val factory = new FutureFactory(executor, stepBarrier)

      try {
        // +1 for the current (scheduler) thread
        stepBarrier.bulkRegister(tasks.length + 1)

        // Only run tasks after all tasks are registered
        val allTasksFuture = factory.runTasks(tasks)

        try {
          while (!stepBarrier.isDone) {
            stepBarrier.await()
            debug("--------tick---------")
            stepClock()
            stepBarrier.completeStep()
          }
        } finally {
          stepBarrier.deRegister()
          stepBarrier.forceTermination()
          Await.result(allTasksFuture, duration.Duration.Inf)
        }
      } catch {
        case e: ExecutionException =>
          val cause = e.getCause()
          if (cause != null)
            throw cause
          else
            throw e
      } finally {
        executor.shutdown();
        executor.awaitTermination(5, TimeUnit.SECONDS);
        executor.shutdownNow()
      }
    }
  }
}
