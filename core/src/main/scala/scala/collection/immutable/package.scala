// SPDX-License-Identifier: Apache-2.0

package scala.collection

package object immutable {
  val LazyList = Stream
  type LazyList[+A] = Stream[A]
}
