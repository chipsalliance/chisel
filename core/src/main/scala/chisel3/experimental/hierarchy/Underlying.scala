// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import chisel3.internal.BaseModule.IsClone

sealed trait Underlying[+T]
final case class Clone[+T](isClone: IsClone[T]) extends Underlying[T]
final case class Proto[+T](proto: T) extends Underlying[T]
