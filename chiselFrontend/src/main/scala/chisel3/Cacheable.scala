package chisel3

import chisel3.experimental.BaseModule
import chisel3.internal.ChiselCacheTag

import scala.reflect.runtime.universe.TypeTag

trait Cacheable[T <: RawModule] extends /*RawModule with*/ Product {
  def tag: ChiselCacheTag = ChiselCacheTag((this.getClass() +: productIterator.toList).hashCode())
  def buildImpl: T
}


object Cache {
  /*
  def createTag[A](parameters: Iterable[A]): Int = {
    (this.getClass() +: parameters.toList).map {
      case c: Cacheable => c.cacheTag
      case o => o.hashCode()
    }.hashCode()
  }
  */
  /*
  def apply[T <: BaseModule](gen: => T, parameters:_*)(implicit tTag: TypeTag[T]): T = {
    val tag = createTag(parameters)
    val dc = chisel3.internal.Builder.

  }
   */
}
