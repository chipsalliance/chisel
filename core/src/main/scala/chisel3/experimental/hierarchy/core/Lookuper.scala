// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}

// Typeclass Trait
trait Lookuper[-V]  {
  type R
  type S
  type G
  def apply[P](value: V, hierarchy: Hierarchy[P]): R
  def setter[P](value: V, lense: Lense[P]): S
  def getter[P](value: V, lense: Lense[P]): G
}
// Typeclass Implementations
object Lookuper {
  
  implicit val lookuperInt = new Lookuper[Int] {
    type R = Int
    type S = Int
    type G = Int
    def setter[P](that: Int, lense: Lense[P]): S = that
    def getter[P](that: Int, lense: Lense[P]): G = that
    def apply[P](that: Int, hierarchy: Hierarchy[P]): R = that
  }
  implicit def lookuperOption[B](implicit lookuper: Lookuper[B]) = new Lookuper[Option[B]] {
    type R = Option[lookuper.R]
    type S = Option[lookuper.S]
    type G = Option[lookuper.G]
    def setter[P](that: Option[B], lense: Lense[P]): S = {
      that.map { x: B => lookuper.setter(x, lense) }
    }
    def getter[P](that: Option[B], lense: Lense[P]): G = {
      that.map { x: B => lookuper.getter(x, lense) }
    }
    def apply[P](that: Option[B], hierarchy: Hierarchy[P]): R = {
      that.map { x: B => lookuper[P](x, hierarchy) }
    }
  }
  implicit def lookuperEither[X, Y](implicit lookuperX: Lookuper[X], lookuperY: Lookuper[Y]) = new Lookuper[Either[X, Y]] {
    type R = Either[lookuperX.R, lookuperY.R]
    type S = Either[lookuperX.S, lookuperY.S]
    type G = Either[lookuperX.G, lookuperY.G]
    def setter[P](that: Either[X, Y], lense: Lense[P]): S = {
      that.map { y: Y => lookuperY.setter(y, lense) }
          .left
          .map { x: X => lookuperX.setter(x, lense) }
    }
    def getter[P](that: Either[X, Y], lense: Lense[P]): G = {
      that.map { y: Y => lookuperY.getter(y, lense) }
          .left
          .map { x: X => lookuperX.getter(x, lense) }
    }
    def apply[P](that: Either[X, Y], hierarchy: Hierarchy[P]): R = {
      that.map { y: Y => lookuperY[P](y, hierarchy) }
          .left
          .map { x: X => lookuperX[P](x, hierarchy) }
    }
  }
  implicit def lookuperTuple2[X, Y](implicit lookuperX: Lookuper[X], lookuperY: Lookuper[Y]) = new Lookuper[Tuple2[X, Y]] {
    type R = Tuple2[lookuperX.R, lookuperY.R]
    type S = Tuple2[lookuperX.S, lookuperY.S]
    type G = Tuple2[lookuperX.G, lookuperY.G]
    def setter[P](that: Tuple2[X, Y], lense: Lense[P]): S = {
      (lookuperX.setter(that._1, lense), lookuperY.setter(that._2, lense))
    }
    def getter[P](that: Tuple2[X, Y], lense: Lense[P]): G = {
      (lookuperX.getter(that._1, lense), lookuperY.getter(that._2, lense))
    }
    def apply[P](that: Tuple2[X, Y], hierarchy: Hierarchy[P]): R = {
      (lookuperX[P](that._1, hierarchy), lookuperY(that._2, hierarchy))
    }
  }
  import scala.language.higherKinds
  implicit def lookuperIterable[B, F[_] <: Iterable[_]](
    implicit lookuper:          Lookuper[B]
  ) = new Lookuper[F[B]] {
    type R = F[lookuper.R]
    type G = F[lookuper.G]
    type S = F[lookuper.S]
    def apply[P](that: F[B], hierarchy: Hierarchy[P]): R = {
      val ret = that.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookuper[P](x, hierarchy) }.asInstanceOf[R]
    }
    def setter[P](that: F[B], lense: Lense[P]): S = {
      val ret = that.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookuper.setter[P](x, lense) }.asInstanceOf[S]
    }
    def getter[P](that: F[B], lense: Lense[P]): G = {
      val ret = that.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookuper.getter[P](x, lense) }.asInstanceOf[G]
    }
  }
  implicit def isLookupable[V <: IsLookupable] = new Lookuper[V] {
    type R = V
    type S = V
    type G = V
    def setter[P](value: V, lense: Lense[P]): V = value
    def getter[P](value: V, lense: Lense[P]): V = value
    def apply[P](value: V, hierarchy: Hierarchy[P]): R = value
  }

  implicit def lookuperContextual[V] = new Lookuper[Contextual[V]] {
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
      println(v)
      val ret = hierarchy.open(v)
      println(ret, hierarchy.proxy.lenses.map(_.edits).mkString(","))
      ret
    }
  }
  implicit def lookupIsInstantiable[U <: IsInstantiable] = new Lookuper[U] {
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
  class SimpleLookuper[V] extends Lookuper[V] {
    type R = V
    type S = V
    type G = V
    def setter[P](that: V, lense: Lense[P]): S = that
    def getter[P](that: V, lense: Lense[P]): G = that
    def apply[P](that: V, hierarchy: Hierarchy[P]): R = that
  }
  implicit val lookuperString = new SimpleLookuper[String]()
}

