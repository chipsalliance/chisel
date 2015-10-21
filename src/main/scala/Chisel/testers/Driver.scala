// See LICENSE for license details.

package Chisel.testers
import Chisel._

object TesterDriver {
  /** For use with modules that should successfully be elaborated by the
    * frontend, and which can be turned into executeables with error codes. */
  def execute(t: => BasicTester): Boolean = {
    val circuit = Builder.build(Module(t))
    //val executable = invokeFIRRTL(circuit)
    //Process(executable) !
    true
  }

  /** For use with modules that should illicit errors from the frontend
    * or which produce IR with consistantly checkable properties. */
  def elaborate(t: => Module): Circuit = {
    Builder.build(Module(t))
  }
}
