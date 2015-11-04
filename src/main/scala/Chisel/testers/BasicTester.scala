// See LICENSE for license details.

package Chisel.testers
import Chisel._

class BasicTester extends Module {
  val io = new Bundle {
    val done = Bool()
    val error = UInt(width = 4)
  }
  io.done := Bool(false)
  io.error := UInt(0)

  def popCount(n: Long) = n.toBinaryString.count(_=='1')
}
