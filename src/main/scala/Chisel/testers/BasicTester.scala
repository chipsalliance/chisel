// See LICENSE for license details.

package Chisel.testers
import Chisel._

import internal._
import internal.Builder.pushCommand
import internal.firrtl._

class BasicTester extends Module {
  // The testbench has no IOs, rather it should communicate using printf, assert, and stop.
  val io = new Bundle()

  def popCount(n: Long): Int = n.toBinaryString.count(_=='1')

  /** Ends the test reporting success.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    */
  def stop() {
    when (!reset) {
      pushCommand(Stop(Node(clock), 0))
    }
  }
}
