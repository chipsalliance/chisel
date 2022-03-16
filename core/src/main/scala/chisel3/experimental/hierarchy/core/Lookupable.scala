// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}

// Typeclass Trait
trait Lookupable[-V] {
  type R
  type S
  type G
  def apply[P](value:  V, hierarchy: Hierarchy[P]): R
  def setter[P](value: V, context:     Context[P]):     S
  def getter[P](value: V, context:     Context[P]):     G
}

// Typeclass Default Implementations
object Lookupable {
  trait SimpleLookupable[V] extends Lookupable[V] {
    type R = V
    type S = V
    type G = V
    def setter[P](value: V, context:     Context[P]):     S = value
    def getter[P](value: V, context:     Context[P]):     G = value
    def apply[P](value:  V, hierarchy: Hierarchy[P]): R = value
  }

  implicit val lookupableInt = new SimpleLookupable[Int] {}
  implicit val lookupableByte = new SimpleLookupable[Byte] {}
  implicit val lookupableShort = new SimpleLookupable[Short] {}
  implicit val lookupableLong = new SimpleLookupable[Long] {}
  implicit val lookupableFloat = new SimpleLookupable[Float] {}
  implicit val lookupableChar = new SimpleLookupable[Char] {}
  implicit val lookupableBoolean = new SimpleLookupable[Boolean] {}
  implicit val lookupableBigInt = new SimpleLookupable[BigInt] {}
  implicit def lookupableOption[B](implicit lookupable: Lookupable[B]) = new Lookupable[Option[B]] {
    type R = Option[lookupable.R]
    type S = Option[lookupable.S]
    type G = Option[lookupable.G]
    def setter[P](value: Option[B], context: Context[P]): S = {
      value.map { x: B => lookupable.setter(x, context) }
    }
    def getter[P](value: Option[B], context: Context[P]): G = {
      value.map { x: B => lookupable.getter(x, context) }
    }
    def apply[P](value: Option[B], hierarchy: Hierarchy[P]): R = {
      value.map { x: B => lookupable[P](x, hierarchy) }
    }
  }
  implicit def lookupableEither[X, Y](implicit lookupableX: Lookupable[X], lookupableY: Lookupable[Y]) =
    new Lookupable[Either[X, Y]] {
      type R = Either[lookupableX.R, lookupableY.R]
      type S = Either[lookupableX.S, lookupableY.S]
      type G = Either[lookupableX.G, lookupableY.G]
      def setter[P](value: Either[X, Y], context: Context[P]): S = {
        value.map { y: Y => lookupableY.setter(y, context) }.left.map { x: X => lookupableX.setter(x, context) }
      }
      def getter[P](value: Either[X, Y], context: Context[P]): G = {
        value.map { y: Y => lookupableY.getter(y, context) }.left.map { x: X => lookupableX.getter(x, context) }
      }
      def apply[P](value: Either[X, Y], hierarchy: Hierarchy[P]): R = {
        value.map { y: Y => lookupableY[P](y, hierarchy) }.left.map { x: X => lookupableX[P](x, hierarchy) }
      }
    }
  implicit def lookupableTuple2[X, Y](implicit lookupableX: Lookupable[X], lookupableY: Lookupable[Y]) =
    new Lookupable[Tuple2[X, Y]] {
      type R = Tuple2[lookupableX.R, lookupableY.R]
      type S = Tuple2[lookupableX.S, lookupableY.S]
      type G = Tuple2[lookupableX.G, lookupableY.G]
      def setter[P](value: Tuple2[X, Y], context: Context[P]): S = {
        (lookupableX.setter(value._1, context), lookupableY.setter(value._2, context))
      }
      def getter[P](value: Tuple2[X, Y], context: Context[P]): G = {
        (lookupableX.getter(value._1, context), lookupableY.getter(value._2, context))
      }
      def apply[P](value: Tuple2[X, Y], hierarchy: Hierarchy[P]): R = {
        (lookupableX[P](value._1, hierarchy), lookupableY(value._2, hierarchy))
      }
    }
  import scala.language.higherKinds
  implicit def lookupableIterable[B, F[_] <: Iterable[_]](
    implicit lookupable: Lookupable[B]
  ) = new Lookupable[F[B]] {
    type R = F[lookupable.R]
    type G = F[lookupable.G]
    type S = F[lookupable.S]
    def apply[P](value: F[B], hierarchy: Hierarchy[P]): R = {
      val ret = value.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable[P](x, hierarchy) }.asInstanceOf[R]
    }
    def setter[P](value: F[B], context: Context[P]): S = {
      val ret = value.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable.setter[P](x, context) }.asInstanceOf[S]
    }
    def getter[P](value: F[B], context: Context[P]): G = {
      val ret = value.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable.getter[P](x, context) }.asInstanceOf[G]
    }
  }
  implicit def isLookupable[V <: IsLookupable] = new Lookupable[V] {
    type R = V
    type S = V
    type G = V
    def setter[P](value: V, context:     Context[P]):     V = value
    def getter[P](value: V, context:     Context[P]):     V = value
    def apply[P](value:  V, hierarchy: Hierarchy[P]): R = value
  }

  implicit def lookupableContextual[V] = new Lookupable[Contextual[V]] {
    type R = V
    type S = ContextualSetter[V]
    type G = Edit[V]
    def setter[P](value: Contextual[V], context: Context[P]): S = {
      ContextualSetter(value, context)
    }
    def getter[P](value: Contextual[V], context: Context[P]): G = {
      context.getEdit(value)
    }
    def apply[P](v: Contextual[V], hierarchy: Hierarchy[P]): V = {
      hierarchy.open(v)
    }
  }
  implicit def lookupIsInstantiable[U <: IsInstantiable] = new Lookupable[U] {
    type R = Instance[U]
    type S = Context[U]
    type G = Context[U]
    def setter[P](value: U, context: Context[P]): S = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[U]].proxy, context.top)
    }
    def getter[P](value: U, context: Context[P]): G = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[U]].proxy, context.top)
    }
    def apply[P](value: U, hierarchy: Hierarchy[P]): Instance[U] = {
      val d = InstantiableDefinition(value)
      val newContexts = hierarchy.proxy.contexts.map { l: Context[P] => l.getter(value)(this).asInstanceOf[Context[U]] }
      val t = InstantiableTransparent(d, newContexts)
      val m = InstantiableMock(t, hierarchy.proxy, Nil)
      m.toInstance
    }
  }
}
