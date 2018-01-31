// See LICENSE for license details.

package chisel3.testers2

import chisel3._

/** Common interface definition for tester backends
  */
trait BackendInterface {
  /** Writes a value to a writable wire.
    * Throws an exception if write is not writable.
    */
  def poke(signal: Data, value: BigInt): Unit

  /** Returns the current combinational value (after any previous pokes have taken effect, and
    * logic has propagated) on a wire.
    */
  def peek(signal: Data): BigInt
  /** Returns the value at the beginning of the current cycle (before any pokes have taken effect) on a wire.
    */
  def stalePeek(signal: Data): BigInt

  def check(signal: Data, value: BigInt): Unit
  def staleCheck(signal: Data, value: BigInt): Unit

  /** Advances the target clock by one cycle.
    */
  def step(signal: Clock, cycles: Int): Unit
}

/** Backend associated with a particular circuit, and can run tests
  */
trait BackendInstance[T <: Module] extends BackendInterface {
  /** Runs of tests are wrapped in this, for any special setup/teardown that needs to happen.
    * Takes the test function, which takes the module used as the testing interface.
    * TesterContext setup is done externally.
    */
  def run(testFn: T => Unit): Unit
}

/** Interface into the testing environment, like ScalaTest
  */
trait TestEnvInterface {
  def testerFail(msg: String): Unit
  def testerExpect(expected: Any, actual: Any, signal: String, msg: Option[String]): Unit
}
