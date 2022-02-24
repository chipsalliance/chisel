// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}

// Marker Trait
trait IsHierarchical

// Wrapper Class
sealed trait Hierarchy[+H] {

  /** Updated by calls to [[_lookup]], to avoid recloning returned Data's */
  private[chisel3] val cache = HashMap[Any, Any]()

  def _lookup[B, C](that: H => B)(
    implicit instancify: Instancify[B],
    h: Contexter[C],
    macroGenerated:  chisel3.internal.MacroGenerated
  ): instancify.R = {
    // TODO: Call to 'that' should be replaced with shapeless to enable deserialized Underlying
    val protoValue = that(this.proto)
    val retValue = instancify(protoValue, this)
    cache.getOrElseUpdate(protoValue, retValue)
    retValue
  }

  /** Determine whether proxy proto is of type provided.
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[B], which will fail if B is an inner class.
    * @note IMPORTANT: this function IGNORES type parameters, akin to normal type erasure.
    * @note IMPORTANT: this function relies on Java reflection for proxy proto, but Scala reflection for provided type
    *
    * E.g. isA[List[Int]] will return true, even if proxy proto is of type List[String]
    * @return Whether proxy proto is of provided type (with caveats outlined above)
    */
  def isA[B: TypeTag]: Boolean = {
    val tptag = implicitly[TypeTag[B]]
    // drop any type information for the comparison, because the proto will not have that information.
    val name = tptag.tpe.toString.takeWhile(_ != '[')
    superClasses.contains(name)
  }

  /** @return Return the proxy Definition[H] of this Hierarchy[H] */
  def toDefinition = Definition(proxy)

  /** @return Convert this Hierarchy[H] as a top-level Instance[H] */
  def toInstance = Instance(proxy)

  private[chisel3] def proxy: Proxy[H]
  private[chisel3] def proto: H = proxy.proto
  private lazy val superClasses = Hierarchy.calculateSuperClasses(proto.getClass())

}

final case class Instance[+H] private[chisel3] (private[chisel3] proxy: Proxy[H]) extends Hierarchy[H]
final case class Definition[+H] private[chisel3] (private[chisel3] proxy: Proxy[H]) extends IsLookupable with Hierarchy[H]

// Typeclass Trait
trait Instancify[V] extends IsTypeclass[V] {
  type R
  def apply[H](value: V, hierarchy: Hierarchy[H]): R
}
// Typeclass Implementations
object Instancify {

  implicit def isLookupable[V <: IsLookupable] = new Instancify[V] {
    type R = V
    def apply[H](value: V, hierarchy: Hierarchy[H]): R = value
  }
  implicit def isContextual[I <: IsContextual](implicit contextualizer: Contextualizer[I]) = new Instancify[I] {
    type R = contextualizer.R 
    def apply[H](value: I, hierarchy: Hierarchy[H]): R = {
      contextualizer(value, hierarchy)
    }
  }
  implicit val instancifyInt = new Instancify[Int] {
    type R = Int
    def apply[H](that: Int, hierarchy: Hierarchy[H]): R = that
  }
  implicit val instancifyString = new Instancify[String] {
    type R = String
    def apply[H](that: String, hierarchy: Hierarchy[H]): R = that
  }
  implicit def instancifyIterable[B, F[_] <: Iterable[_]](
    implicit instancify:          Instancify[B]
  ) = new Instancify[F[B]] {
    type R = F[instancify.R]
    def apply[H](that: F[B], hierarchy: Hierarchy[H]): R = {
      val ret = that.asInstanceOf[Iterable[B]]
      ret.map { x: B => instancify[H](x, hierarchy) }.asInstanceOf[R]
    }
  }
  implicit def instancifyOption[B](implicit instancify: Instancify[B]) = new Instancify[Option[B]] {
    type R = Option[instancify.R]
    def apply[H](that: Option[B], hierarchy: Hierarchy[H]): R = {
      that.map { x: B => instancify[H](x, hierarchy) }
    }
  }
  implicit def instancifyEither[X, Y](implicit instancifyX: Instancify[X], instancifyY: Instancify[Y]) = new Instancify[Either[X, Y]] {
    type R = Either[instancifyX.R, instancifyY.R]
    def apply[H](that: Either[X, Y], hierarchy: Hierarchy[H]): R = {
      that.map { y: Y => instancifyY[H](y, hierarchy) }
          .left
          .map { x: X => instancifyX[H](x, hierarchy) }
    }
  }
  implicit def instancifyTuple2[X, Y](implicit instancifyX: Instancify[X], instancifyY: Instancify[Y]) = new Instancify[Tuple2[X, Y]] {
    type R = Tuple2[instancifyX.R, instancifyY.R]
    def apply[H](that: Tuple2[X, Y], hierarchy: Hierarchy[H]): R = {
      (instancifyX[H](that._1, hierarchy), instancifyY(that._2, hierarchy))
    }
  }
}

object Definition {
  def apply[T](proto: => T): Definition[T] =
    macro DefinitionTransform.apply[T]
  def do_apply[T](proto: => T)(implicit buildable: Buildable[T]): Definition[T] = {
    Definition(buildable(proto))
  }

}

object Instance {
  def apply[T](definition: Definition[T]): Instance[T] =
    macro InstanceTransform.apply[T]
  def do_apply[T](definition: Definition[T])(implicit stampable: Stampable[T]): Instance[T] = {
    Instance(stampable(definition))
  }
}


object Hierarchy {

  // This code handles a special-case where, within an mdoc context, the type returned from
  //  scala reflection (typetag) looks different than when returned from java reflection.
  //  This function detects this case and reshapes the string to match.
  private def modifyReplString(clz: String): String = {
    if (clz != null) {
      clz.split('.').toList match {
        case "repl" :: "MdocSession" :: app :: rest => s"$app.this." + rest.mkString(".")
        case other                                  => clz
      }
    } else clz
  }
  // Nested objects stick a '$' at the end of the object name, but this does not show up in the scala reflection type string
  // E.g.
  // object Foo {
  //   object Bar {
  //     class Baz() 
  //   }
  // }
  // Scala type will be Foo.Bar.Baz
  // Java type will be Foo.Bar$.Baz
  private def modifyNestedObjects(clz: String): String = {
    if (clz != null) { clz.replace("$","") } else clz
  }
  private def calculateSuperClasses(clz: Class[_]): Set[String] = {
    if (clz != null) {
      Set(modifyNestedObjects(modifyReplString(clz.getCanonicalName()))) ++
        clz.getInterfaces().flatMap(i => calculateSuperClasses(i)) ++
        calculateSuperClasses(clz.getSuperclass())
    } else {
      Set.empty[String]
    }
  }
}