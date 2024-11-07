// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

/** While this is public, it is not recommended for users to extend directly.
  * Instead, use the [[instantiable]] annotation on your trait or class.
  *
  * This trait indicates whether a class can be returned from an Instance.
  */
trait IsInstantiable

object IsInstantiable {
  implicit class IsInstantiableExtensions[T <: IsInstantiable](i: T) {
    @deprecated(
      "Use of @instantiable on user-defined types is deprecated. Implement Lookupable for your type instead.",
      "Chisel 7.0.0"
    )
    def toInstance: Instance[T] = new Instance(Proto(i))
  }
}
