// SPDX-License-Identifier: Apache-2.0

package chisel3
package simulator

import scala.concurrent._
import scala.util.DynamicVariable
import java.util.concurrent.Phaser

private[simulator] final class StepBarrier(debug: Boolean = true) {
  final val preStep:  Phaser = new Phaser(0)
  final val postStep: Phaser = new Phaser(0)

  def debugPrintln(x: => Any): Unit = {
    if (debug) println(s"[Thr:${Thread.currentThread().threadId()}] $x")
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
    debugPrintln("step")
    preStep.arriveAndAwaitAdvance()
    postStep.arriveAndAwaitAdvance()
  }

  def await(): Unit = {
    debugPrintln("--await--")
    preStep.arriveAndAwaitAdvance()
  }

  def completeStep(): Unit = {
    postStep.arriveAndAwaitAdvance()
    debugPrintln("---completeStep---\n")
  }

  def forceTermination(): Unit = {
    preStep.forceTermination()
    postStep.forceTermination()
  }
}

private[simulator] object StepBarrier {
  private val dynamicVariable = new DynamicVariable[Option[StepBarrier]](None)

  def withValue[T](stepBarrier: => StepBarrier)(body: => T): T =
    dynamicVariable.withValue(Some(stepBarrier))(body)

  def currentOption: Option[StepBarrier] = dynamicVariable.value
}
