// See LICENSE for license details.

package firrtl.transforms

import firrtl._
import firrtl.annotations._

/** Marks modules as "must deduplicate" */
case class MustDeduplicateAnnotation(modules: Seq[IsModule]) extends Annotation {

  def update(renames: RenameMap): Seq[MustDeduplicateAnnotation] = {
    val newModules: Seq[IsModule] = modules.flatMap { m =>
      renames.get(m) match {
        case None        => Seq(m)
        case Some(Seq()) => Seq()
        case Some(Seq(one: IsModule)) => Seq(one)
        case Some(many) =>
          val msg = "Something went wrong! This anno's targets should only rename to IsModules! " +
            s"Got: ${m.serialize} -> ${many.map(_.serialize).mkString(", ")}"
          throw new Exception(msg)
      }
    }
    if (newModules.isEmpty) Seq() else Seq(this.copy(newModules))
  }
}
