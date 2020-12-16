// SPDX-License-Identifier: Apache-2.0

package firrtl.annotations

import firrtl._
import firrtl.analyses.InstanceKeyGraph
import firrtl.analyses.InstanceKeyGraph.InstanceKey
import firrtl.annotations.TargetToken._

object TargetUtils {

  /** Turns an instance path into a corresponding [[IsModule]]
    *
    * @note First InstanceKey is treated as the [[CircuitTarget]]
    * @param path Instance path
    * @param start Module in instance path to be starting [[ModuleTarget]]
    * @return [[IsModule]] corresponding to Instance path
    */
  def instKeyPathToTarget(path: Seq[InstanceKey], start: Option[String] = None): IsModule = {
    val head = path.head
    val startx = start.getOrElse(head.module)
    val top: IsModule = CircuitTarget(head.module).module(startx) // ~Top|Start
    val pathx = path.dropWhile(_.module != startx)
    if (pathx.isEmpty) top
    else pathx.tail.foldLeft(top) { case (acc, key) => acc.instOf(key.name, key.module) }
  }

  /** Calculates all [[InstanceTarget]]s that refer to the given [[IsModule]]
    *
    * {{{
    * ~Top|Top/a:A/b:B/c:C unfolds to:
    * * ~Top|Top/a:A/b:B/c:C
    * * ~Top|A/b:B/c:C
    * * ~Top|B/c:C
    * }}}
    * @note [[ModuleTarget]] arguments return an empty Iterable
    */
  def unfoldInstanceTargets(ismod: IsModule): Iterable[InstanceTarget] = {
    // concretely use List which is fast in practice
    def rec(im: IsModule): List[InstanceTarget] = im match {
      case inst: InstanceTarget => inst :: rec(inst.stripHierarchy(1))
      case _ => Nil
    }
    rec(ismod)
  }
}
