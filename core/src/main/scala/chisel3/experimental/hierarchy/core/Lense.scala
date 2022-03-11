package chisel3.experimental.hierarchy.core
import scala.collection.mutable
import java.util.IdentityHashMap

// Wrapper Class
//final case class Lense[+P](proxy: Proxy[P]) {
//
//  private[chisel3] val cache = new IdentityHashMap[Any, Any]()
//
//  def _lookup[B, C](that: P => B)(
//    implicit lenser: Lenser[B],
//    macroGenerated:  chisel3.internal.MacroGenerated
//  ): lenser.R = {
//    val protoValue = that(this.proxy.proto)
//    val retValue = lenser(protoValue, this)
//    //cache.getOrElseUpdate(protoValue, retValue)
//    retValue
//  }
//}

//final case class Socket[T](contextual: Contextual[T], lense: Lense[_]) {
//  def value: T = contextual.value
//  def value_=(newValue: T): Edit[T] = edit({_: T => newValue})
//  def edit(f: T => T): Edit[T] = lense.proxy match {
//    case i: HierarchicalProxy[_, _] => i.addEdit(contextual, f)
//    case i: InstantiableProxy[_, _] => i.parent match {
//      case i: HierarchicalProxy[_, _] => i.addEdit(contextual, f)
//    }
//    case other => chisel3.internal.throwException("Cannot edit on a non-IsContext lense!")
//  }
//}

// Underlying Classes; For now, just use IsContext's until proven otherwise
//
// Typeclass Trait
//trait Lenser[V]  {
//  type R
//  def apply[P](value: V, lense: Lense[P]): R
//}
//
//object Lenser {
//  implicit def instance[I](implicit lookuper: Lookuper[Instance[I]]) = new Lenser[Instance[I]] {
//    type R = Lense[I]
//    def apply[P](value: Instance[I], lense: Lense[P]): R = {
//      // For now i'm just wrapping it in instance, i don't think it matters as of now, but something to investigate later
//      Lense(lookuper(value, lense.proxy).asInstanceOf[Instance[I]].proxy)
//    }
//  }
//  implicit def contextual[I] = new Lenser[Contextual[I]] {
//    type R = Socket[I]
//    def apply[P](value: Contextual[I], lense: Lense[P]): R = {
//      Socket(value, lense)
//    }
//  }
//  //implicit def isContextual[V <: IsContextual](implicit lookuper: Lookuper[V]) = new Lenser[V] {
//  //  type R = lookuper.R
//  //  def apply[P](v: V, lense: Lense[P]): R = {
//  //    lookuper(v, lense.toHierarchy)
//  //  }
//  //}
//  //implicit def isContext[V <: IsContext](implicit lookuper: Lookuper[V]) = new Lenser[V] {
//  //  type R = Lense[V]
//  //  def apply[P](value: V, lense: Lense[P]): R = {
//  //    // For now i'm just wrapping it in instance, i don't think it matters as of now, but something to investigate later
//  //    Lense(lookuper(value, lense.toHierarchy).asInstanceOf[Hierarchy[V]].proxy)
//  //  }
//  //}
//  implicit def isInstantiable[V <: IsInstantiable](implicit lookuper: Lookuper[V]) = new Lenser[V] {
//    type R = Lense[V]
//    def apply[P](value: V, lense: Lense[P]): R = {
//      Lense(lookuper(value, lense.proxy).asInstanceOf[Hierarchy[V]].proxy)
//    }
//  }
//  implicit def isLookupable[I <: IsLookupable] = new Lenser[I] {
//    type R = I
//    def apply[P](value: I, lense: Lense[P]): I = value
//  }
//  implicit val lenserInt = new Lenser[Int] {
//    type R = Int
//    def apply[P](that: Int, lense: Lense[P]): R = that
//  }
//  implicit val lenserString = new Lenser[String] {
//    type R = String
//    def apply[P](that: String, lense: Lense[P]): R = that
//  }
//  //implicit def lenserIterable[B, F[_] <: Iterable[_]](
//  //  implicit lenser: Lenser[B]
//  //) = new Lenser[F[B]] {
//  //  type R = F[lenser.R]
//  //  def apply[P](that: F[B], lense: Lense[P]): R = {
//  //    val ret = that.asInstanceOf[Iterable[B]]
//  //    ret.map { x: B => lenser[P](x, lense) }.asInstanceOf[R]
//  //  }
//  //}
//  implicit def lenserOption[B](implicit lenser: Lenser[B]) = new Lenser[Option[B]] {
//    type R = Option[lenser.R]
//    def apply[P](that: Option[B], lense: Lense[P]): R = {
//      that.map { x: B => lenser[P](x, lense) }
//    }
//  }
//  implicit def lenserEither[X, Y](implicit lenserX: Lenser[X], lenserY: Lenser[Y]) = new Lenser[Either[X, Y]] {
//    type R = Either[lenserX.R, lenserY.R]
//    def apply[P](that: Either[X, Y], lense: Lense[P]): R = {
//      that.map { y: Y => lenserY[P](y, lense) }
//          .left
//          .map { x: X => lenserX[P](x, lense) }
//    }
//  }
//  implicit def lenserTuple2[X, Y](implicit lenserX: Lenser[X], lenserY: Lenser[Y]) = new Lenser[Tuple2[X, Y]] {
//    type R = Tuple2[lenserX.R, lenserY.R]
//    def apply[P](that: Tuple2[X, Y], lense: Lense[P]): R = {
//      (lenserX[P](that._1, lense), lenserY(that._2, lense))
//    }
//  }
//}
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