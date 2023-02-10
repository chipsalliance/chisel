// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.ir._

/** Flattens Bundles and Vecs.
  * - Some implicit bundle types remain, but with a limited depth:
  *   - the type of a memory is still a bundle with depth 2 (mem -> port -> field), see [[MemPortUtils.memType]]
  *   - the type of a module instance is still a bundle with depth 1 (instance -> port)
  */
object LowerTypes {

  /** Delimiter used in lowering names */
  val delim = "_"

  /** Expands a chain of referential [[firrtl.ir.Expression]]s into the equivalent lowered name
    * @param e [[firrtl.ir.Expression]] made up of _only_ [[firrtl.WRef]], [[firrtl.WSubField]], and [[firrtl.WSubIndex]]
    * @return Lowered name of e
    * @note Please make sure that there will be no name collisions when you use this outside of the context of LowerTypes!
    */
  def loweredName(e: Expression): String = e match {
    case e: Reference => e.name
    case e: SubField  => s"${loweredName(e.expr)}$delim${e.name}"
    case e: SubIndex  => s"${loweredName(e.expr)}$delim${e.value}"
  }
  def loweredName(s: Seq[String]): String = s.mkString(delim)

}
