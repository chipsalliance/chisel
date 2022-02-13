// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

/** While this is public, it is not recommended for users to extend directly.
  * Instead, use the [[@instantiable]] annotation on your trait or class.
  *
  * This trait indicates whether a class can be returned from an Instance.
  */
trait IsInstantiable

object IsInstantiable {
  implicit class IsInstantiableExtensions[T <: IsInstantiable](i: T) {
    //TODO: Not sure an empty context is correct here. In fact its probably wrong.
    def toInstance: Instance[T] = new Instance(Proto(i), Contexts())
  }
}
