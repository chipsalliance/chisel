// SPDX-License-Identifier: Apache-2.0

package chisel3

/** The util package provides extensions to core chisel for common hardware components and utility
  * functions
  */
package object util {

  /** Synonyms, moved from main package object - maintain scope. */
  type ValidIO[+T <: Data] = chisel3.util.Valid[T]
  val ValidIO = chisel3.util.Valid
  val DecoupledIO = chisel3.util.Decoupled
}
