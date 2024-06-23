//> using repository "sonatype-s01:snapshots"
//> using scala "2.13.14"
//> using dep "org.chipsalliance::chisel:latest.integration"
//> using plugin "org.chipsalliance:::chisel-plugin:latest.integration"
//> using options "-unchecked", "-deprecation", "-language:reflectiveCalls", "-feature", "-Xcheckinit", "-Ymacro-annotations"
import java.util.Random

// import chisel3._
// // _root_ disambiguate from package chisel3.util.circt if user imports chisel3.util._
// import _root_.circt.stage.ChiselStage
import scala.concurrent._
import scala.concurrent.duration._
import java.util.concurrent.{CyclicBarrier, Phaser}


// Worker function
// import scala.concurrent.ExecutionContext.Implicits.global

object LockStepExample extends App {

  val stepBarrier = new StepBarrier()

  val tasks = (1 to 33).map { id =>
    Task(
      { () =>
        for (i <- 1 to (2220 + scala.util.Random.nextInt(100))) {
          println(s"Worker $id: trying step $i")
          System.out.flush()
          stepBarrier.step()
        }
      // println(s"Worker $id: de-registering")
      },
      id
    )
  }

  val scheduler = new Scheduler(stepBarrier, tasks)(scala.concurrent.ExecutionContext.global)
  scheduler.run()

}

LockStepExample.main(Array.empty)
