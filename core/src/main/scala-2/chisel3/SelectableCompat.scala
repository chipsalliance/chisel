// SPDX-License-Identifier: Apache-2.0

package chisel3

/** Shim for Scala 3's `scala.reflect.Selectable`
  *
  * Allows Chisel to cross-compile Scala 2 and Scala 3
  * while enabling structural typing in Scala 3.
  */
trait ReflectSelectable

/** Shim for Scala 3's `scala.Selectable`
  *
  * Allows Chisel to cross-compile Scala 2 and Scala 3
  * while enabling structural typing in Scala 3.
  */
trait Selectable
