// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.incremental.{Cache, Stash, StashOptions}
import chisel3.stage.ChiselGeneratorAnnotation
import firrtl.AnnotationSeq
import firrtl.options.{Phase, PreservesAll}

import scala.collection.mutable

/** Consume all Cache annotations and return a Stash annotation
  */
class FillStash extends Phase with PreservesAll[Phase] {

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val caches = mutable.ArrayBuffer[Cache]()
    val gens = mutable.ArrayBuffer[ChiselGeneratorAnnotation]()
    val retAnnotations = annotations.flatMap {
      case c: Cache => caches += c; None
      case g: ChiselGeneratorAnnotation => gens += g; None
      case other => Some(other)
    }
    val useLatests = annotations.collect {
      case stash: StashOptions => stash.useLatest
    }.distinct

    require(useLatests.size <= 1, s"Multiple stash behaviors set!!")
    val useLatest = useLatests.headOption.getOrElse(false)

    val cacheMap = caches.map { c => (c.packge -> c ) }.toMap
    gens.map {
      case g@ChiselGeneratorAnnotation(_, None) => g.copy(stashOpt = Some(Stash(cacheMap, useLatest)))
      case g => g
    } ++ retAnnotations
  }
}
