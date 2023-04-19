// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

/** While this is public, it is not recommended for users to extend directly.
  * Instead, use the [[instantiable]] annotation on your trait or class.
  *
  * This trait indicates whether a class can be returned from an Instance.
  */
trait IsInstantiable {

  /** Used to record which val's are marked @public, so we can ensure all IO's are @public'd */
  def _all_at_public: Option[Seq[Any]] = None
  def _clock_reset:   Seq[Any] = Nil
}

object IsInstantiable {
  implicit class IsInstantiableExtensions[T <: IsInstantiable](i: T) {
    def toInstance: Instance[T] = new Instance(Proto(i))
  }
}
