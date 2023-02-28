// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

package object plugin {

  // The actual implementation
  private def _autoNameRecursively[T <: Any](prevId: Long, name: String, nameMe: T): T = {
    chisel3.internal.Builder.nameRecursively(
      name,
      nameMe,
      (id: chisel3.internal.HasId, n: String) => {
        // Name override only if result was created in this scope
        if (id._id > prevId) {
          id.forceAutoSeed(n)
        }
      }
    )
    nameMe
  }

  /** Used by Chisel's compiler plugin to automatically name signals
    * DO NOT USE in your normal Chisel code!!!
    *
    * @param name The name to use
    * @param nameMe The thing to be named
    * @tparam T The type of the thing to be named
    * @return The thing, but now named
    */
  def autoNameRecursively[T <: Any](name: String)(nameMe: => T): T = {
    // The _id of the most recently constructed HasId
    val prevId = Builder.idGen.value
    val result = nameMe
    _autoNameRecursively(prevId, name, result)
  }

  /** Used by Chisel's compiler plugin to automatically name signals
    * DO NOT USE in your normal Chisel code!!!
    *
    * @param names The names to use corresponding to interesting fields of the Product
    * @param nameMe The [[scala.Product]] to be named
    * @tparam T The type of the thing to be named
    * @return The thing, but with each member named
    */
  def autoNameRecursivelyProduct[T <: Product](names: List[Option[String]])(nameMe: => T): T = {
    // The _id of the most recently constructed HasId
    val prevId = Builder.idGen.value
    val result = nameMe
    for ((name, t) <- names.iterator.zip(result.productIterator) if name.nonEmpty) {
      _autoNameRecursively(prevId, name.get, t)
    }
    result
  }
}
