// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}

// Marker Trait
trait IsHierarchicable

// Wrapper Class
sealed trait Hierarchy[+A] {

  /** Updated by calls to [[_lookup]], to avoid recloning returned Data's */
  private[chisel3] val cache = HashMap[Any, Any]()

  def _lookup[B, C](that: A => B)(
    implicit instancify: Instancify[B],
    h: Hierarchicable[A],
    macroGenerated:  chisel3.internal.MacroGenerated
  ): instancify.C = {
    // TODO: Call to 'that' should be replaced with shapeless to enable deserialized Underlying
    val protoValue = that(this.proto)
    val retValue = instancify(protoValue, this)
    cache.getOrElseUpdate(protoValue, retValue)
    retValue
  }

  /** Determine whether underlying proto is of type provided.
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[B], which will fail if B is an inner class.
    * @note IMPORTANT: this function IGNORES type parameters, akin to normal type erasure.
    * @note IMPORTANT: this function relies on Java reflection for underlying proto, but Scala reflection for provided type
    *
    * E.g. isA[List[Int]] will return true, even if underlying proto is of type List[String]
    * @return Whether underlying proto is of provided type (with caveats outlined above)
    */
  def isA[B: TypeTag]: Boolean = {
    val tptag = implicitly[TypeTag[B]]
    // drop any type information for the comparison, because the proto will not have that information.
    val name = tptag.tpe.toString.takeWhile(_ != '[')
    superClasses.contains(name)
  }

  /** @return Return the underlying Definition[A] of this Hierarchy[A] */
  def toDefinition = Definition(underlying)

  /** @return Convert this Hierarchy[A] as a top-level Instance[A] */
  def toInstance = Instance(underlying)

  private[chisel3] def underlying: Underlying[A]
  private[chisel3] def proto: A = underlying.proto
  private lazy val superClasses = Hierarchy.calculateSuperClasses(proto.getClass())

}

final case class Instance[+A] private[chisel3] (private[chisel3] underlying: Underlying[A]) extends Hierarchy[A]
final case class Definition[+A] private[chisel3] (private[chisel3] underlying: Underlying[A]) extends IsLookupable with Hierarchy[A]

// Typeclass Trait
trait Instancify[B] extends IsTypeclass[B] {
  type C
  def apply[A](b: B, context: Hierarchy[A])(implicit h: Hierarchicable[A]): C
}
// Typeclass Implementations
object Instancify {

  implicit def isLookupable[L <: IsLookupable] = new Instancify[L] {
    type C = L
    def apply[A](b: L, context: Hierarchy[A])(implicit h: Hierarchicable[A]): L = b
  }
  implicit def isContextual[I <: IsContextual](implicit contextualize: Contextualize[I]) = new Instancify[I] {
    type C = contextualize.C 
    def apply[A](b: I, context: Hierarchy[A])(implicit h: Hierarchicable[A]): C = {
      contextualize(b, context)
    }
  }
  implicit val instancifyInt = new Instancify[Int] {
    type C = Int
    def apply[A](that: Int, context: Hierarchy[A])(implicit h: Hierarchicable[A]): C = that
  }
  implicit val instancifyString = new Instancify[String] {
    type C = String
    def apply[A](that: String, context: Hierarchy[A])(implicit h: Hierarchicable[A]): C = that
  }
  implicit def instancifyIterable[B, F[_] <: Iterable[_]](
    implicit instancify:          Instancify[B]
  ) = new Instancify[F[B]] {
    type C = F[instancify.C]
    def apply[A](that: F[B], context: Hierarchy[A])(implicit h: Hierarchicable[A]): C = {
      val ret = that.asInstanceOf[Iterable[B]]
      ret.map { x: B => instancify[A](x, context) }.asInstanceOf[C]
    }
  }
  implicit def instancifyOption[B](implicit instancify: Instancify[B]) = new Instancify[Option[B]] {
    type C = Option[instancify.C]
    def apply[A](that: Option[B], context: Hierarchy[A])(implicit h: Hierarchicable[A]): C = {
      that.map { x: B => instancify[A](x, context) }
    }
  }
  implicit def instancifyEither[L, R](implicit instancifyL: Instancify[L], instancifyR: Instancify[R]) = new Instancify[Either[L, R]] {
    type C = Either[instancifyL.C, instancifyR.C]
    def apply[A](that: Either[L, R], context: Hierarchy[A])(implicit h: Hierarchicable[A]): C = {
      that.map { x: R => instancifyR[A](x, context) }
          .left
          .map { x: L => instancifyL[A](x, context) }
    }
  }
  implicit def instancifyTuple2[X, Y](implicit instancifyX: Instancify[X], instancifyY: Instancify[Y]) = new Instancify[Tuple2[X, Y]] {
    type C = Tuple2[instancifyX.C, instancifyY.C]
    def apply[A](that: Tuple2[X, Y], context: Hierarchy[A])(implicit h: Hierarchicable[A]): C = {
      (instancifyX[A](that._1, context), instancifyY(that._2, context))
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