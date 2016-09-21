// See LICENSE for license details.

package chisel3

package object util {

  /** Synonyms, moved from main package object - maintain scope. */
  type ValidIO[+T <: Data] = chisel3.util.Valid[T]
  val ValidIO = chisel3.util.Valid
  val DecoupledIO = chisel3.util.Decoupled

}
