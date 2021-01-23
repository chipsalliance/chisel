// SPDX-License-Identifier: Apache-2.0

package scala.collection

import scala.collection.immutable.ListMap

package object immutable {
  val SeqMap = ListMap
  type SeqMap[K, +V] = ListMap[K, V]

  val VectorMap = ListMap
  type VectorMap[K, +V] = ListMap[K, V]
}
