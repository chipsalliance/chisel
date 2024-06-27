package chisel3
package simulator

import scala.concurrent._
import scala.util.DynamicVariable

import java.util.concurrent.ForkJoinPool

case class Task(runnable: () => Unit, id: Int)

// If needed we can use fancier Executors, e.g. ThreadPoolExecutor
class ExecutorWithDynamicVariable[T](dynamicVariable: DynamicVariable[T]) extends ForkJoinPool {
  override def execute(task: Runnable): Unit = {
    val copiedValue = dynamicVariable.value
    super.execute(new Runnable {
      override def run = {
        dynamicVariable.value = copiedValue
        task.run
      }
    })
  }
}

object SynchContext {
  private val dynamicVariable = new DynamicVariable[Option[StepBarrier]](None)
  def withSynchBarrier[T](body: => T): T = {
    dynamicVariable.withValue(Some(new StepBarrier()))(body)
  }

  def currentOption: Option[StepBarrier] = dynamicVariable.value

  implicit val executionContext: ExecutionContext = scala.concurrent.ExecutionContext.fromExecutor(
    new ExecutorWithDynamicVariable(dynamicVariable)
  )
}

private[simulator] final class Scheduler(val stepClock: () => Unit) {

  def run(tasks: Seq[Task]) = {
    SynchContext.withSynchBarrier {
      val stepBarrier = synchronized { SynchContext.currentOption.get }
      import SynchContext.executionContext

      def worker(task: Task): Future[Unit] = Future {
        try {
          task.runnable()
        } finally {
          stepBarrier.deRegister()
        }
      }

      try {
        stepBarrier.bulkRegister(tasks.length + 1)

        val workerFutures = tasks.map(worker)

        try {
          while (!stepBarrier.isDone) {
            stepBarrier.await()
            stepClock()
            stepBarrier.completeStep()
          }
          stepBarrier.forceTermination()
        } catch {
          case e: java.util.concurrent.ExecutionException =>
            val cause = e.getCause()
            if (cause != null)
              throw cause
            else
              throw e
        } finally {
          stepBarrier.deRegister()
          stepBarrier.forceTermination()
          Await.result(Future.sequence(workerFutures), duration.Duration.Inf)
        }
      } catch {
        case e: Throwable =>
          stepBarrier.forceTermination()
          throw e
      }
    }
  }

}
