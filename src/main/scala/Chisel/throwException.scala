// See LICENSE for license details.

package Chisel

@deprecated("throwException doesn't exist in Chisel3", "3.0.0")
@throws(classOf[Exception])
object throwException {
  def apply(s: String, t: Throwable = null) = {
    val xcpt = new Exception(s, t)
    throw xcpt
  }
}
