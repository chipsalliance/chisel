// See LICENSE for license details.

package chisel.compatibility

import java.io.File

import chisel._

@deprecated("chiselMain doesn't exist in Chisel3", "3.0") object chiselMain {
  def apply[T <: Module](args: Array[String], gen: () => T): Unit =
    Predef.assert(false, "No more chiselMain in Chisel3")

  def run[T <: Module] (args: Array[String], gen: () => T): Unit = {
    val circuit = Driver.elaborate(gen)
    Driver.parseArgs(args)
    val output_file = new File(Driver.targetDir + "/" + circuit.name + ".fir")
    Driver.dumpFirrtl(circuit, Option(output_file))
  }
}
