// See LICENSE for license details.

package chisel3.tester

import scala.collection.mutable

import org.scalatest._
import org.scalatest.exceptions.TestFailedException

import chisel3._
import chisel3.experimental.MultiIOModule

trait ChiselScalatestTester extends Assertions with TestEnvInterface {
  protected val batchedFailures = mutable.ArrayBuffer[TestFailedException]()

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
    if (expected != actual) {
      val appendMsg = msg match {
        case Some(msg) => s": $msg"
        case _ => ""
      }
      batchedFailures += new TestFailedException(s"$signal=$actual did not equal expected=$expected$appendMsg", 5)
    }
  }

  override def checkpoint(): Unit = {
    // TODO: report multiple exceptions simultaneously
    for (failure <- batchedFailures) {
      throw failure
    }
  }

  override def test[T <: MultiIOModule](tester: BackendInstance[T])(testFn: T => Unit) {
    batchedFailures.clear()
    Context.run(tester, this, testFn)
  }
}
