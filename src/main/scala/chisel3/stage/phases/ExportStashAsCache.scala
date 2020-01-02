// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.incremental.{ExportCache, Stash, StashOptions}
import firrtl.AnnotationSeq
import firrtl.options.{Phase, PreservesAll}

import scala.collection.mutable

/** Export stashed elaborations as a cache
  */
class ExportStashAsCache extends Phase with PreservesAll[Phase] {

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    var stashOpt: Option[Stash] = None
    val exportOptions = mutable.ArrayBuffer[ExportCache]()
    val retAnnotations = annotations.flatMap {
      case s: Stash => stashOpt = Some(s); None
      case s: ExportCache => exportOptions += s; None
      case other => Some(other)
    }
    require(stashOpt.nonEmpty, "Cannot export stash to cache without a stash!")
    val stash = stashOpt.get

    val caches = exportOptions.map {
      case ExportCache(packge, backingDir, isFat) =>
       stash.exportAsCache(packge, backingDir, isFat)
    }
    caches ++ retAnnotations
  }

}
