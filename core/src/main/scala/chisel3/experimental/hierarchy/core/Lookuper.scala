// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}

// Typeclass Trait
trait Lookuper[V]  {
  type R
  def apply[P](value: V, hierarchy: Hierarchy[P]): R
}
// Typeclass Implementations
object Lookuper {

  implicit def isLookupable[V <: IsLookupable] = new Lookuper[V] {
    type R = V
    def apply[P](value: V, hierarchy: Hierarchy[P]): R = value
  }
  implicit def isContextual[I <: IsContextual](implicit contextualizer: Contextualizer[I]) = new Lookuper[I] {
    type R = contextualizer.R 
    def apply[P](value: I, hierarchy: Hierarchy[P]): R = {
      contextualizer(value, hierarchy)
    }
  }
  implicit val lookuperInt = new Lookuper[Int] {
    type R = Int
    def apply[P](that: Int, hierarchy: Hierarchy[P]): R = that
  }
  implicit val lookuperString = new Lookuper[String] {
    type R = String
    def apply[P](that: String, hierarchy: Hierarchy[P]): R = that
  }
  implicit def lookuperIterable[B, F[_] <: Iterable[_]](
    implicit lookuper:          Lookuper[B]
  ) = new Lookuper[F[B]] {
    type R = F[lookuper.R]
    def apply[P](that: F[B], hierarchy: Hierarchy[P]): R = {
      val ret = that.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookuper[P](x, hierarchy) }.asInstanceOf[R]
    }
  }
  implicit def lookuperOption[B](implicit lookuper: Lookuper[B]) = new Lookuper[Option[B]] {
    type R = Option[lookuper.R]
    def apply[P](that: Option[B], hierarchy: Hierarchy[P]): R = {
      that.map { x: B => lookuper[P](x, hierarchy) }
    }
  }
  implicit def lookuperEither[X, Y](implicit lookuperX: Lookuper[X], lookuperY: Lookuper[Y]) = new Lookuper[Either[X, Y]] {
    type R = Either[lookuperX.R, lookuperY.R]
    def apply[P](that: Either[X, Y], hierarchy: Hierarchy[P]): R = {
      that.map { y: Y => lookuperY[P](y, hierarchy) }
          .left
          .map { x: X => lookuperX[P](x, hierarchy) }
    }
  }
  implicit def lookuperTuple2[X, Y](implicit lookuperX: Lookuper[X], lookuperY: Lookuper[Y]) = new Lookuper[Tuple2[X, Y]] {
    type R = Tuple2[lookuperX.R, lookuperY.R]
    def apply[P](that: Tuple2[X, Y], hierarchy: Hierarchy[P]): R = {
      (lookuperX[P](that._1, hierarchy), lookuperY(that._2, hierarchy))
    }
  }
}

