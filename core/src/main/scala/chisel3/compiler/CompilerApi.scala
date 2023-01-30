// SPDX-License-Identifier: Apache-2.0

package chisel3.compiler

/** Context of current compiler.
  * the chisel3 package only provide [[firtool]] compiler.
  * add 'chisel3-compiler-panama' or 'chisel3-compiler-jni' to use [[panama]] or [[jni]]
  */
private[chisel3] trait CompilerApi {
  /** Consume a [[chisel3.internal.firrtl.Circuit]], feed to compiler
    * @todo in the future, we may need to add [[chisel3.internal.Builder]] here
    *       for Chisel Language Server
    */
  def consume(circuit: chisel3.internal.firrtl.Circuit): Unit

  /** Add option compiler. */
  def setOption(option: String): Unit

  /** Get current configured options.
    * @return the sequence of options
    */
  def getOptions: Seq[String]

  /** Ask Compiler to emit files based on options. */
  def emit: Unit
}