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
  def setter[P](value: V, lense:     Lense[P]):     S
  def getter[P](value: V, lense:     Lense[P]):     G
}

// Typeclass Default Implementations
object Lookupable {
  trait SimpleLookupable[V] extends Lookupable[V] {
    type R = V
    type S = V
    type G = V
    def setter[P](value: V, lense:     Lense[P]):     S = value
    def getter[P](value: V, lense:     Lense[P]):     G = value
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
    def setter[P](value: Option[B], lense: Lense[P]): S = {
      value.map { x: B => lookupable.setter(x, lense) }
    }
    def getter[P](value: Option[B], lense: Lense[P]): G = {
      value.map { x: B => lookupable.getter(x, lense) }
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
      def setter[P](value: Either[X, Y], lense: Lense[P]): S = {
        value.map { y: Y => lookupableY.setter(y, lense) }.left.map { x: X => lookupableX.setter(x, lense) }
      }
      def getter[P](value: Either[X, Y], lense: Lense[P]): G = {
        value.map { y: Y => lookupableY.getter(y, lense) }.left.map { x: X => lookupableX.getter(x, lense) }
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
      def setter[P](value: Tuple2[X, Y], lense: Lense[P]): S = {
        (lookupableX.setter(value._1, lense), lookupableY.setter(value._2, lense))
      }
      def getter[P](value: Tuple2[X, Y], lense: Lense[P]): G = {
        (lookupableX.getter(value._1, lense), lookupableY.getter(value._2, lense))
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
    def setter[P](value: F[B], lense: Lense[P]): S = {
      val ret = value.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable.setter[P](x, lense) }.asInstanceOf[S]
    }
    def getter[P](value: F[B], lense: Lense[P]): G = {
      val ret = value.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable.getter[P](x, lense) }.asInstanceOf[G]
    }
  }
  implicit def isLookupable[V <: IsLookupable] = new Lookupable[V] {
    type R = V
    type S = V
    type G = V
    def setter[P](value: V, lense:     Lense[P]):     V = value
    def getter[P](value: V, lense:     Lense[P]):     V = value
    def apply[P](value:  V, hierarchy: Hierarchy[P]): R = value
  }

  implicit def lookupableContextual[V] = new Lookupable[Contextual[V]] {
    type R = V
    type S = Socket[V]
    type G = Edit[V]
    def setter[P](value: Contextual[V], lense: Lense[P]): S = {
      Socket(value, lense)
    }
    def getter[P](value: Contextual[V], lense: Lense[P]): G = {
      lense.getEdit(value)
    }
    def apply[P](v: Contextual[V], hierarchy: Hierarchy[P]): V = {
      hierarchy.open(v)
    }
  }
  implicit def lookupIsInstantiable[U <: IsInstantiable] = new Lookupable[U] {
    type R = Instance[U]
    type S = Lense[U]
    type G = Lense[U]
    def setter[P](value: U, lense: Lense[P]): S = {
      NestedLense(apply(value, lense.toHierarchy).asInstanceOf[Instance[U]].proxy, lense.top)
    }
    def getter[P](value: U, lense: Lense[P]): G = {
      NestedLense(apply(value, lense.toHierarchy).asInstanceOf[Instance[U]].proxy, lense.top)
    }
    def apply[P](value: U, hierarchy: Hierarchy[P]): Instance[U] = {
      val d = InstantiableDefinition(value)
      val newLenses = hierarchy.proxy.lenses.map { l: Lense[P] => l.getter(value)(this).asInstanceOf[Lense[U]] }
      val t = InstantiableTransparent(d, newLenses)
      val m = InstantiableMock(t, hierarchy.proxy, Nil)
      m.toInstance
    }
  }
}
