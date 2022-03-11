package chisel3.experimental.hierarchy.core
import scala.collection.mutable
import java.util.IdentityHashMap

// Wrapper Class
trait Lense[+P] {
  def proxy: Proxy[P]
  def top: TopLense[_]
  def toHierarchy: Hierarchy[P] = proxy match {
    case d: DefinitionProxy[P] => Definition(d)
    case i: InstanceProxy[P] => Instance(i)
  }
  private[chisel3] val edits = new IdentityHashMap[Contextual[Any], Any => Any]()
  private[chisel3] val cache = new IdentityHashMap[Any, Any]()

  private[chisel3] def addEdit[T](contextual: Contextual[T], edit: T => T): Unit = {
    require(!edits.containsKey(contextual), s"Cannot set $contextual more than once!")
    edits.put(contextual, edit.asInstanceOf[Any => Any])
  }
  private[chisel3] def getEdit[T](contextual: Contextual[T]): Edit[T] = {
    if(edits.containsKey(contextual)) Edit(contextual, edits.get(contextual).asInstanceOf[T => T]) else {
      Edit(contextual, {t: T => t})
    }
  }
  private[chisel3] def compute[T](key: Contextual[T], c: Contextual[T]): Contextual[T] = new Contextual(getEdit(key).edit(c.value))

  def _lookup[B, C](that: P => B)(
    implicit lookuper: Lookuper[B],
    macroGenerated:  chisel3.internal.MacroGenerated
  ): lookuper.S = {
    val protoValue = that(this.proxy.proto)
    if(cache.containsKey(protoValue)) cache.get(protoValue).asInstanceOf[lookuper.S] else {
      val retValue = lookuper.setter(protoValue, this)
      cache.put(protoValue, retValue)
      retValue
    }
  }
  def getter[B, C](protoValue: B)(
    implicit lookuper: Lookuper[B]
  ): lookuper.G = {
    if(cache.containsKey(protoValue)) cache.get(protoValue).asInstanceOf[lookuper.G] else {
      println(protoValue)
      val retValue = lookuper.getter(protoValue, this)
      cache.put(protoValue, retValue)
      println(retValue)
      retValue
    }
  }
}
final case class TopLense[+P](proxy: Proxy[P]) extends Lense[P] {
  def top = this
}
final case class NestedLense[+P](proxy: Proxy[P], top: TopLense[_]) extends Lense[P] {

}

final case class Socket[T](contextual: Contextual[T], lense: Lense[_]) {
  def value: T = contextual.value
  def value_=(newValue: T): TopLense[_] = edit({_: T => newValue})
  def edit(f: T => T): TopLense[_] = {
    lense.addEdit(contextual, f)
    lense.top
  }
}
final case class Edit[T](contextual: Contextual[T], edit: T => T)

// Underlying Classes; For now, just use IsContext's until proven otherwise
//
// Typeclass Trait
//trait Lenser[V]  {
//}
//object Lenser {
//  implicit def instance[I](implicit lookuper: Lookuper[Instance[I]]) = new Lenser[Instance[I]] {
//  }
//}
//  //implicit def lenserIterable[B, F[_] <: Iterable[_]](
//  //  implicit lenser: Lenser[B]
//  //) = new Lenser[F[B]] {
//  //  type R = F[lenser.R]
//  //  def apply[P](that: F[B], lense: Lense[P]): R = {
//  //    val ret = that.asInstanceOf[Iterable[B]]
//  //    ret.map { x: B => lenser[P](x, lense) }.asInstanceOf[R]
//  //  }
//  //}
//
//
//
////TODO: will need to handle lensing nested Contextuals with another typeclass
//// TODO: nested contextuals
////final case class ContextualLense[T, V](value: V, parent: Proxy[T, IsContext])
//  // TODO: nested contextuals
//  //implicit def isContextual[I <: IsContextual] = new Lensify[Contextual[I], Edit[I]] {
//  //  def lensify[C](b: Contextual[I], hierarchy: Lense[C]): Edit[I] = Edit(b.value)
//  //}