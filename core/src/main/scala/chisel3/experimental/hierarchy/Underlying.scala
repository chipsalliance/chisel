// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

/** Represents the underlying implementation of a Definition or Instance */
sealed trait Underlying[+T <: IsInstantiable] {
  def contexts: Contexts
  //def withContext(pf: PartialFunction[Any, Any]): Underlying[T]
}

/** A clone of a real implementation */
final case class Clone[+T <: IsInstantiable](isClone: IsClone[T]) extends Underlying[T] {
  def contexts = isClone.contexts
  //def withContext(pf: PartialFunction[Any, Any]): Underlying[T] = Clone(isClone.withContext(pf))
}

/** An actual implementation */
final case class Proto[+T <: IsInstantiable](proto: T) extends Underlying[T] {
  def contexts = Contexts()
  //def withContext(pf: PartialFunction[Any, Any]): Underlying[T] = ??? //Not sure what to do here
}
