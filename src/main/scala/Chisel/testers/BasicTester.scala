// See LICENSE for license details.

package Chisel.testers
import Chisel._

import internal._
import internal.Builder.pushCommand
import firrtl._

class BasicTester extends Module {
  val io = new Bundle {
    val done = Bool()
    val error = UInt(width = 4)
  }
  io.done := Bool(false)
  io.error := UInt(0)

  def popCount(n: Long): Int = n.toBinaryString.count(_=='1')

  /** Ends the test, reporting success.
    */
  def stop() {
    pushCommand(Stop(Node(clock), 0))
  }
}
