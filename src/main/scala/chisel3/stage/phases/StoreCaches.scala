// See LICENSE for license details.

package chisel3.stage.phases

import chisel3.incremental.Cache
import firrtl.AnnotationSeq
import firrtl.options.{Phase, PreservesAll}

import scala.collection.mutable

/** Consume all Cache annotations and return a Stash annotation
  */
class StoreCaches extends Phase with PreservesAll[Phase] {

  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val caches = mutable.ArrayBuffer[Cache]()
    val retAnnotations = annotations.flatMap {
      case c: Cache => caches += c; None
      case other => Some(other)
    }

    caches.foreach {
      case cache if cache.backingDirectory.nonEmpty => cache.writeTo(cache.backingDirectory.get)
    }

    retAnnotations
  }

}
