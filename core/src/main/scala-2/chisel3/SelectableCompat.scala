// SPDX-License-Identifier: Apache-2.0

package chisel3

/** These traits do nothing in Scala 2 other than maintaining
  * compatibility with Scala 3's Selectable and
  * scala.reflect.Selectable, required to support structural selection
  * in Scala 3.
  *
  * SelectableCompat => scala.reflect.Selectable
  * Selectable => scala.Selectable
  */
trait SelectableCompat
trait Selectable
