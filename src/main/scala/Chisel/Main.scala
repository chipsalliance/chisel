// See LICENSE for details

package Chisel

@deprecated("chiselMain doesn't exist in Chisel3", "3.0") object chiselMain {
  def apply[T <: Module](args: Array[String], gen: () => T) =
    Predef.assert(false)
}
