package chisel3
package simulator

import chisel3.util._

import scala.concurrent._
import java.util.concurrent.Phaser

private[simulator] final class StepBarrier() {
  final val preStep:  Phaser = new Phaser(0)
  final val postStep: Phaser = new Phaser(0)

  def register(): Int = synchronized {
    assert(!preStep.isTerminated())
    preStep.register()
    assert(!postStep.isTerminated())
    postStep.register()
  }

  def bulkRegister(n: Int): Int = {
    assert(!preStep.isTerminated(), s"preStep is terminated")
    preStep.bulkRegister(n)
    assert(!postStep.isTerminated(), "postStep is terminated")
    postStep.bulkRegister(n)
  }

  def isDone: Boolean = {
    preStep.getRegisteredParties() <= 1 || postStep
      .getRegisteredParties() <= 1 || preStep.isTerminated || postStep.isTerminated
  }

  def deRegister(): Unit = {
    preStep.awaitAdvance(preStep.arriveAndDeregister())
    postStep.awaitAdvance(postStep.arriveAndDeregister())
  }

  def step(): Unit = {
    preStep.arriveAndAwaitAdvance()
    postStep.arriveAndAwaitAdvance()
  }

  def completeStep(): Unit = {
    postStep.arriveAndAwaitAdvance()
  }

  def await(): Unit = {
    preStep.arriveAndAwaitAdvance()
  }

  def forceTermination(): Unit = {
    preStep.forceTermination()
    postStep.forceTermination()
  }
}
