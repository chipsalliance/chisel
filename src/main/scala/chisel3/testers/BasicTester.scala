// See LICENSE for license details.

package chisel3.testers
import chisel3._

import scala.language.experimental.macros

import internal._
import internal.Builder.pushCommand
import internal.firrtl._
import internal.sourceinfo.SourceInfo
//import chisel3.core.ExplicitCompileOptions.NotStrict

class BasicTester extends Module() {
  // The testbench has no IOs, rather it should communicate using printf, assert, and stop.
  val io = IO(new Bundle() {})

  def popCount(n: Long): Int = n.toBinaryString.count(_=='1')

  /** Ends the test reporting success.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    */
  def stop()(implicit sourceInfo: SourceInfo) {
    // TODO: rewrite this using library-style SourceInfo passing.
    when (!reset.toBool) {
      pushCommand(Stop(sourceInfo, Node(clock), 0))
    }
  }

  /** The finish method provides a hook that subclasses of BasicTester can use to
    * alter a circuit after their constructor has been called.
    * For example, a specialized tester subclassing BasicTester could override finish in order to
    * add flow control logic for a decoupled io port of a device under test
    */
  def finish(): Unit = {}
}
