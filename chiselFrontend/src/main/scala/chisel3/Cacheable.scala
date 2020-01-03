package chisel3

import chisel3.experimental.BaseModule
import chisel3.incremental.ItemTag

trait Cacheable extends BaseModule with Product {
  def tag = ItemTag[this.type](productIterator.toList)
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
