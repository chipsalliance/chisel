package chisel3
package simulator

trait ThreadedChiselSimAPI extends ChiselSimAPI {

  sealed class ForkBuilder(
    scheduler: Scheduler,
    tasks:     Seq[Task] = Seq.empty) {

    def fork(runnable: => Unit): ForkBuilder =
      new ForkBuilder(scheduler, tasks :+ Task(() => runnable, tasks.length + 1))

    def joinAndStep(): Unit = {
      join()
      scheduler.stepClock()
    }

    def join(): Unit = {
      scheduler.run(tasks)
    }
  }

  object fork {

    private def stepClock(cycles: Int = 1): Unit = {
      val module = AnySimulatedModule.current
      val clock = DutContext.current.clock.get
      module.willEvaluate()
      if (cycles == 0) {
        module.controller.run(0)
      } else {
        val simulationPort = module.port(clock)
        simulationPort.tick(
          timestepsPerPhase = 1,
          cycles = cycles,
          inPhaseValue = 0,
          outOfPhaseValue = 1
        )
      }
    }

    def apply(runnable: => Unit): ForkBuilder =
      new ForkBuilder(new Scheduler(() => stepClock())).fork(runnable)
  }

  implicit class threadedTestableClock(clock: Clock) extends testableClock(clock) {
    override def step(cycles: Int = 1): Unit = {
      for (_ <- 0 until cycles) {
        SynchContext.currentOption match {
          case Some(barrier) =>
            barrier.step()
          case None =>
            super.step()
        }
      }
    }

    override def stepUntil(sentinelPort: Data, sentinelValue: BigInt, maxCycles: Int): Unit = {
      for (_ <- 0 until maxCycles) {
        step()
        if (sentinelPort.peekValue().asBigInt == sentinelValue) {
          return
        }
      }
    }
  }

  override protected def getDutClock: testableClock = {
    threadedTestableClock(
      // TODO: handle clock not being present
      DutContext.current.clock.get
    )
  }
}
