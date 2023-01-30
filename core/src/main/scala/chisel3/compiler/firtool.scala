// SPDX-License-Identifier: Apache-2.0

package chisel3.compiler

object firtool {
  def apply: CompilerApi = new FirtoolImpl
}

private[chisel3] class FirtoolImpl extends CompilerApi {
  private var _consumed: Boolean = false
  private var _circuit: chisel3.internal.firrtl.Circuit = null
  private val _options: scala.collection.mutable.ArrayBuffer[String] = scala.collection.mutable.ArrayBuffer()

  /** Consume a [[chisel3.internal.firrtl.Circuit]], feed to compiler */
  def consume(circuit: chisel3.internal.firrtl.Circuit): Unit = {
    if (_consumed) {
      throw new chisel3.ChiselException("Already consume a circuit in this firrtl instance.")
    } else {
      _circuit = circuit
    }
    _consumed = true
  }

  /** Add option compiler. */
  final def setOption(option: String): Unit = {
    require(option != "--annotation-file", "annotation-file is produced by Chisel")
    _options += option
  }

  final def emit: Unit = os.proc(
    "firtool",
    _options ++ Some(s"--annotation-file=${os.temp(firrtl.annotations.JsonProtocol.serialize(_circuit.firrtlAnnotations.toSeq))}"),
    os.temp(chisel3.internal.firrtl.Converter.convertLazily(_circuit).serialize)
  ).call()

  /** Get current configured options.
    *
    * @return the sequence of options in pair
    */
  override def getOptions: Seq[String] = _options
}
