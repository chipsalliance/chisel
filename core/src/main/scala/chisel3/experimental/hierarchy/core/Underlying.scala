// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import chisel3.internal.BaseModule.IsClone

/** Represents the underlying implementation of a Definition or Instance */
sealed trait Underlying[+T]

/** A clone of a real implementation */
final case class Clone[+T](isClone: IsClone[T]) extends Underlying[T]

/** An actual implementation */
final case class Proto[+T](proto: T) extends Underlying[T]
