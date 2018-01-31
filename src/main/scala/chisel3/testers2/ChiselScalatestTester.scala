// See LICENSE for license details.

package chisel3.testers2

import org.scalatest._
import org.scalatest.exceptions.TestFailedException

import chisel3._

trait ChiselScalatestTester extends Assertions with TestEnvInterface {
  // Stack trace depths:
  // 0: this function
  // 1: TestEnvInterface.testerExpect (superclass of this)
  // 2: BackendInterface.check
  // 3: (implicit testable*).check
  // 4: user code calling check

  def testerFail(msg: String): Unit = {
    throw new TestFailedException(s"$msg", 4)
  }

  def testerExpect(expected: Any, actual: Any, signal: String, msg: Option[String]): Unit = {
    if (expected != actual) {
      val appendMsg = msg match {
        case Some(msg) => s": $msg"
        case _ => ""
      }
      throw new TestFailedException(s"$signal=$actual did not equal expected=$expected$appendMsg", 4)
    }
  }

  def test[T <: Module](tester: BackendInstance[T])(testFn: T => Unit) {
    tester.run(dut => {
      Context.run(tester, this, testFn)
    })
  }
}
