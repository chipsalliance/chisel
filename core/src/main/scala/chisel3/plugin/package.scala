package chisel3

package object plugin {

  // Used by Chisel's compiler plugin to automatically name signals
  def autoNameRecursively[T <: Any](name: String, nameMe: T): T = {
    chisel3.internal.Builder.nameRecursively(
      name.replace(" ", ""),
      nameMe,
      (id: chisel3.internal.HasId, n: String) => id.autoSeed(n)
    )
    nameMe
  }

}
