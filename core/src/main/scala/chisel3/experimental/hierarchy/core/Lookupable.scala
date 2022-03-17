// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import scala.language.higherKinds // Necessary for lookupableIterable

/** Typeclass describing the return result (and type) when looking up a value of type V from a Hierarchy[P] or Context[P]
  *
  * Default implementations are contained in the companion object
  */
trait Lookupable[-V] {

  /** Return type when looking up a value of type V from a Hierarchy */
  type R

  /** Result when looking up a value of type V
    *
    * @param value @public value contained in the underlying proto
    * @param hierarchy where the lookup is occuring
    * @return value as looked up from hierarchy
    */
  def apply[P](value: V, hierarchy: Hierarchy[P]): R

  /** Return type when looking up a value of type V, with the intention of setting a contextual from a Context */
  type S

  /** Result when looking up a value of type V, with the intention of setting a contextual from a Context
    *
    * @param value @public value contained in the underlying proto
    * @param context where the lookup is occuring
    * @return value as looked up from context
    */
  def setter[P](value: V, context: Context[P]): S

  /** Return type when looking up a value of type V, with the intention of getting a contextual from a Context */
  type G

  /** Result when looking up a value of type V, with the intention of getting a contextual from a Context
    *
    * @param value @public value contained in the underlying proto
    * @param context where the lookup is occuring
    * @return value as looked up from context
    */
  def getter[P](value: V, context: Context[P]): G
}

/** Default implementations of Lookupable */
object Lookupable {

  /** Simple Lookupable implementation where the value is always returned, without modification */
  trait SimpleLookupable[V] extends Lookupable[V] {
    override type R = V
    override type S = V
    override type G = V
    override def setter[P](value: V, context:   Context[P]):   S = value
    override def getter[P](value: V, context:   Context[P]):   G = value
    override def apply[P](value:  V, hierarchy: Hierarchy[P]): R = value
  }

  // Lookups for primitive and simple objects

  implicit val lookupableInt = new SimpleLookupable[Int] {}
  implicit val lookupableByte = new SimpleLookupable[Byte] {}
  implicit val lookupableShort = new SimpleLookupable[Short] {}
  implicit val lookupableLong = new SimpleLookupable[Long] {}
  implicit val lookupableFloat = new SimpleLookupable[Float] {}
  implicit val lookupableChar = new SimpleLookupable[Char] {}
  implicit val lookupableBoolean = new SimpleLookupable[Boolean] {}
  implicit val lookupableBigInt = new SimpleLookupable[BigInt] {}

  // Lookups for simple containers of other values
  implicit def lookupableOption[B](implicit lookupable: Lookupable[B]) = new Lookupable[Option[B]] {
    override type R = Option[lookupable.R]
    override def apply[P](value: Option[B], hierarchy: Hierarchy[P]): R = value.map { x: B =>
      lookupable[P](x, hierarchy)
    }

    override type S = Option[lookupable.S]
    override def setter[P](value: Option[B], context: Context[P]): S = value.map { x: B =>
      lookupable.setter(x, context)
    }

    override type G = Option[lookupable.G]
    override def getter[P](value: Option[B], context: Context[P]): G = value.map { x: B =>
      lookupable.getter(x, context)
    }
  }

  implicit def lookupableEither[X, Y](implicit lookupableX: Lookupable[X], lookupableY: Lookupable[Y]) =
    new Lookupable[Either[X, Y]] {
      override type R = Either[lookupableX.R, lookupableY.R]
      override def apply[P](value: Either[X, Y], hierarchy: Hierarchy[P]): R = value.map { y: Y =>
        lookupableY[P](y, hierarchy)
      }.left.map { x: X => lookupableX[P](x, hierarchy) }

      override type S = Either[lookupableX.S, lookupableY.S]
      def setter[P](value: Either[X, Y], context: Context[P]): S = value.map { y: Y =>
        lookupableY.setter(y, context)
      }.left.map { x: X => lookupableX.setter(x, context) }

      override type G = Either[lookupableX.G, lookupableY.G]
      def getter[P](value: Either[X, Y], context: Context[P]): G = value.map { y: Y =>
        lookupableY.getter(y, context)
      }.left.map { x: X => lookupableX.getter(x, context) }
    }

  implicit def lookupableTuple2[X, Y](implicit lookupableX: Lookupable[X], lookupableY: Lookupable[Y]) =
    new Lookupable[Tuple2[X, Y]] {
      override type R = Tuple2[lookupableX.R, lookupableY.R]
      override def apply[P](value: Tuple2[X, Y], hierarchy: Hierarchy[P]): R =
        (lookupableX[P](value._1, hierarchy), lookupableY(value._2, hierarchy))

      override type S = Tuple2[lookupableX.S, lookupableY.S]
      override def setter[P](value: Tuple2[X, Y], context: Context[P]): S =
        (lookupableX.setter(value._1, context), lookupableY.setter(value._2, context))

      override type G = Tuple2[lookupableX.G, lookupableY.G]
      override def getter[P](value: Tuple2[X, Y], context: Context[P]): G =
        (lookupableX.getter(value._1, context), lookupableY.getter(value._2, context))
    }

  implicit def lookupableIterable[B, F[_] <: Iterable[_]](
    implicit lookupable: Lookupable[B]
  ) = new Lookupable[F[B]] {
    override type R = F[lookupable.R]
    override def apply[P](value: F[B], hierarchy: Hierarchy[P]): R = {
      val ret = value.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable[P](x, hierarchy) }.asInstanceOf[R]
    }

    override type S = F[lookupable.S]
    override def setter[P](value: F[B], context: Context[P]): S = {
      val ret = value.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable.setter[P](x, context) }.asInstanceOf[S]
    }

    override type G = F[lookupable.G]
    override def getter[P](value: F[B], context: Context[P]): G = {
      val ret = value.asInstanceOf[Iterable[B]]
      ret.map { x: B => lookupable.getter[P](x, context) }.asInstanceOf[G]
    }
  }

  // Lookups for hierarchy.core objects
  implicit def lookupableContextual[V] = new Lookupable[Contextual[V]] {
    override type R = V
    override def apply[P](v: Contextual[V], hierarchy: Hierarchy[P]): V = hierarchy.open(v)

    override type S = ContextualSetter[V]
    override def setter[P](value: Contextual[V], context: Context[P]): S = ContextualSetter(value, context)

    override type G = Edit[V]
    override def getter[P](value: Contextual[V], context: Context[P]): G = context.getEdit(value)
  }

  implicit def isLookupable[V <: IsLookupable] = new SimpleLookupable[V] {}

  implicit def lookupIsInstantiable[U <: IsInstantiable] = new Lookupable[U] {
    override type R = Instance[U]
    override def apply[P](value: U, hierarchy: Hierarchy[P]): Instance[U] = {
      val d = InstantiableDefinition(value)
      val newContexts = hierarchy.proxy.contexts.map { l: Context[P] => l.getter(value)(this).asInstanceOf[Context[U]] }
      val t = InstantiableTransparent(d, newContexts)
      val m = InstantiableMock(t, hierarchy.proxy, Nil)
      m.toInstance
    }

    override type S = Context[U]
    override def getter[P](value: U, context: Context[P]): G = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[U]].proxy, context.root)
    }

    override type G = Context[U]
    override def setter[P](value: U, context: Context[P]): S = {
      NestedContext(apply(value, context.toHierarchy).asInstanceOf[Instance[U]].proxy, context.root)
    }
  }
}
