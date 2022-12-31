// SPDX-License-Identifier: Apache-2.0

package chisel3.compiler
import chisel3.ChiselException
import os.Path

import scala.collection.mutable.ArrayBuffer

object firtool {
  def apply: CompilerApi  = new FirtoolImpl
}

class FirtoolImpl extends CompilerApi {
  /** Consume a [[chisel3.internal.firrtl.Circuit]], feed to compiler */
  def consume(circuit: chisel3.internal.firrtl.Circuit): Unit = {
    if(_consumed) {
      throw new ChiselException("")
    } else {
      ???
    }
    _consumed = true
  }

  /** Add option compiler. */
  final def setOption(option: String): Unit = _options += option

  /** Get current configured options. */
  final def getOptions: Seq[String] = _options.toSeq

  /** Ask Compiler to emit files based on options. */
  override def emit: Path = ???

  private var _consumed: Boolean = false
  private val _options: scala.collection.mutable.ArrayBuffer[String] = ArrayBuffer()
}
