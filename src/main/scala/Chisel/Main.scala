// See LICENSE for license details.

package Chisel

import java.io.File

@deprecated("chiselMain doesn't exist in Chisel3", "3.0") object chiselMain {
  def apply[T <: Module](args: Array[String], gen: () => T): Unit =
    Predef.assert(false)

  def run[T <: Module] (args: Array[String], gen: () => T): Unit = {
    def circuit = Driver.elaborate(gen)
    def output_file = new File(Driver.targetDir + "/" + circuit.name + ".fir")
    Driver.parseArgs(args)
    Driver.dumpFirrtl(circuit, Option(output_file))
  }
}
