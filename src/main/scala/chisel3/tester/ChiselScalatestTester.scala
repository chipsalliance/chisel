// See LICENSE for license details.

package chisel3.tester

import scala.collection.mutable

import org.scalatest._
import org.scalatest.exceptions.TestFailedException

import chisel3._
import chisel3.experimental.MultiIOModule

trait ChiselScalatestTester extends Assertions with TestEnvInterface {
  protected val batchedFailures = mutable.ArrayBuffer[TestFailedException]()

  // Stack trace data to help generate more informative (and localizable) failure messages
  protected var topFileName: Option[String] = None  // best guess at the testdriver top filename

  // Stack trace depths:
  // 0: this function
  // 1: TestEnvInterface.testerExpect (superclass of this)
  // 2: BackendInterface.expect
  // 3: (implicit testable*).expectWithStale
  // 4: (implicit testable*).expect
  // 5: user code calling check
  override def testerFail(msg: String): Unit = {
    batchedFailures += new TestFailedException(s"$msg", 4)
  }

  override def testerExpect(expected: Any, actual: Any, signal: String, msg: Option[String]): Unit = {
    val callingStackDepth = 5
    if (expected != actual) {
      val appendMsg = msg match {
        case Some(msg) => s": $msg"
        case _ => ""
      }
      // TODO: this depends on all user test code being in a new thread (so much of the plumbing
      // is pre-filtered out) - which is true only for the ThreadedBackend.
      val detailedTrace = topFileName.map { fileName =>
        // Add 4 to the depth:
        // 3 for the map and applies
        // 1 for the element that will be reported by ScalaTest
        val lineNumbers = (new Throwable).getStackTrace.drop(callingStackDepth + 4).collect {
          case ste if ste.getFileName == fileName => ste.getLineNumber
        }.mkString(", ")
        if (lineNumbers.isEmpty()) {
          ""
        } else {
          s" (lines in $fileName: $lineNumbers)"
        }
      }.getOrElse("")
      batchedFailures += new TestFailedException(
          s"$signal=$actual did not equal expected=$expected$appendMsg$detailedTrace",
          callingStackDepth)
    }
  }

  override def checkpoint(): Unit = {
    // TODO: report multiple exceptions simultaneously
    for (failure <- batchedFailures) {
      throw failure
    }
  }

  override def test[T <: MultiIOModule](tester: BackendInstance[T])(testFn: T => Unit) {
    // Try and get the user's top-level test filename
    val internalFiles = Set("ChiselScalatestTester.scala", "BackendInterface.scala")
    val topFileNameGuess = (new Throwable).getStackTrace.apply(1).getFileName()
    if (internalFiles.contains(topFileNameGuess)) {
      println("Unable to guess top-level testdriver filename from stack trace")
      topFileName = None
    } else {
      topFileName = Some(topFileNameGuess)
    }

    batchedFailures.clear()
    Context.run(tester, this, testFn)
  }
}
