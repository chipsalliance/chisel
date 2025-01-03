// SPDX-License-Identifier: Apache-2.0

package chisel3
package simulator

import chisel3.experimental.SourceInfo

trait ThreadedChiselSimAPI extends ChiselSimAPI {

  sealed class ForkBuilder(
    scheduler: Scheduler,
    tasks:     Seq[Task] = Seq.empty) {

    def fork(runnable: => Unit): ForkBuilder =
      new ForkBuilder(scheduler, tasks :+ Task(() => runnable, tasks.length + 1))

    def join(): Unit =
      scheduler.run(tasks)

    def joinAndStep(): Unit = {
      join()
      scheduler.stepClock()
    }
  }

  object fork {
    val currentModule = AnySimulatedModule.current
    val clock = DutContext.current.clock.get
    val clockPort = currentModule.port(clock)

    private def tickClock(): Unit = {
      clockPort.tick(
        timestepsPerPhase = 1,
        cycles = 1,
        inPhaseValue = 0,
        outOfPhaseValue = 1
      )
    }

    def apply(runnable: => Unit): ForkBuilder =
      new ForkBuilder(new Scheduler(() => tickClock())).fork(runnable)
  }

  implicit class threadedTestableClock(clock: Clock)(implicit val sourceInfo: SourceInfo) extends testableClock(clock) {
    override def step(cycles: Int = 1): Unit = {
      StepBarrier.currentOption match {
        case Some(barrier) =>
          for (_ <- 0 until cycles) {
            barrier.step()
          }
        case None =>
          super.step(cycles)
      }
    }

    override def stepUntil(sentinelPort: Data, sentinelValue: BigInt, maxCycles: Int): Unit = {
      StepBarrier.currentOption match {
        case Some(barrier) =>
          for (_ <- 0 until maxCycles) {
            if (sentinelPort.peekValue().asBigInt == sentinelValue) {
              return
            }
            barrier.step()
          }
        case None =>
          super.stepUntil(sentinelPort, sentinelValue, maxCycles)
      }
    }
  }

  override protected def currentClock: Option[testableClock] =
    DutContext.current.clock.map(threadedTestableClock(_))

}
