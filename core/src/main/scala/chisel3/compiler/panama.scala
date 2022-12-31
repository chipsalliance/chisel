// SPDX-License-Identifier: Apache-2.0

package chisel3.compiler

object panama {
  def apply: CompilerApi = Class.forName("chisel3.compiler.PanamaImpl").getConstructor().newInstance().asInstanceOf[CompilerApi]
}
