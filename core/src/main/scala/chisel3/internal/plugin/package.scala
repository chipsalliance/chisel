// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

package object plugin {
  /** Used by Chisel's compiler plugin to automatically name signals
    * DO NOT USE in your normal Chisel code!!!
    *
    * @param name The name to use
    * @param nameMe The thing to be named
    * @tparam T The type of the thing to be named
    * @return The thing, but now named
    */
  def autoNameRecursively[T <: Any](name: String, nameMe: T): T = {
    chisel3.internal.Builder.nameRecursively(
      name.replace(" ", ""),
      nameMe,
      (id: chisel3.internal.HasId, n: String) => id.autoSeed(n)
    )
    nameMe
  }
}
