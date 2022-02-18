// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core
import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}

// Marker Trait
trait IsHierarchicable {
  def parent: Option[IsHierarchicable]
  def toUnderlying: Underlying[IsHierarchicable] = this match {
    case i: IsStandIn[IsHierarchicable] => StandIn(i)
    case o => Proto(o, parent.map(_.toUnderlying))
  }
}

// Wrapper Class
sealed trait Hierarchy[+A] {

  /** Updated by calls to [[_lookup]], to avoid recloning returned Data's */
  private[chisel3] val cache = HashMap[Any, Any]()

  def _lookup[B, C](that: A => B)(
    implicit instancify: Instancify[B],
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
  def apply[A](b: B, context: Hierarchy[A]): C
}

// Typeclass Implementations
object Instancify {
  implicit def isLookupable[L <: IsLookupable] = new Instancify[L] {
    type C = L
    def apply[A](b: L, context: Hierarchy[A]): L = b
  }
  implicit def isContextual[I <: IsContextual](implicit contextualize: Contextualize[I]) = new Instancify[I] {
    type C = contextualize.C 
    def apply[A](b: I, context: Hierarchy[A]): C = {
      contextualize(b, context)
    }
  }
  //implicit def isOther[X](implicit contextualize: Contextualize[X]) = new Instancify[X] {
  //  type C = contextualize.C
  //  def apply[A](b: X, context: Hierarchy[A]): C = {
  //    new Instance(contextualize(b, context).asInstan)
  //  }
  //}
//  import scala.language.higherKinds // Required to avoid warning for lookupIterable type parameter
//  implicit def lookupIterable[B, F[_] <: Iterable[_]](
//    implicit sourceInfo: SourceInfo,
//    compileOptions:      CompileOptions,
//    lookupable:          Lookupable[B]
//  ) = new Lookupable[F[B]] {
//    type C = F[lookupable.C]
//    def definitionLookup[A](that: A => F[B], definition: Definition[A]): C = {
//      val ret = that(definition.proto).asInstanceOf[Iterable[B]]
//      ret.map { x: B => lookupable.definitionLookup[A](_ => x, definition) }.asInstanceOf[C]
//    }
//    def instanceLookup[A](that: A => F[B], instance: Instance[A]): C = {
//      import instance._
//      val ret = that(proto).asInstanceOf[Iterable[B]]
//      ret.map { x: B => lookupable.instanceLookup[A](_ => x, instance) }.asInstanceOf[C]
//    }
//  }
//  implicit def lookupOption[B](
//    implicit sourceInfo: SourceInfo,
//    compileOptions:      CompileOptions,
//    lookupable:          Lookupable[B]
//  ) = new Lookupable[Option[B]] {
//    type C = Option[lookupable.C]
//    def definitionLookup[A](that: A => Option[B], definition: Definition[A]): C = {
//      val ret = that(definition.proto)
//      ret.map { x: B => lookupable.definitionLookup[A](_ => x, definition) }
//    }
//    def instanceLookup[A](that: A => Option[B], instance: Instance[A]): C = {
//      import instance._
//      val ret = that(proto)
//      ret.map { x: B => lookupable.instanceLookup[A](_ => x, instance) }
//    }
//  }
//  implicit def lookupEither[L, R](
//    implicit sourceInfo: SourceInfo,
//    compileOptions:      CompileOptions,
//    lookupableL:         Lookupable[L],
//    lookupableR:         Lookupable[R]
//  ) = new Lookupable[Either[L, R]] {
//    type C = Either[lookupableL.C, lookupableR.C]
//    def definitionLookup[A](that: A => Either[L, R], definition: Definition[A]): C = {
//      val ret = that(definition.proto)
//      ret.map { x: R => lookupableR.definitionLookup[A](_ => x, definition) }.left.map { x: L =>
//        lookupableL.definitionLookup[A](_ => x, definition)
//      }
//    }
//    def instanceLookup[A](that: A => Either[L, R], instance: Instance[A]): C = {
//      import instance._
//      val ret = that(proto)
//      ret.map { x: R => lookupableR.instanceLookup[A](_ => x, instance) }.left.map { x: L =>
//        lookupableL.instanceLookup[A](_ => x, instance)
//      }
//    }
//  }
//
//  implicit def lookupTuple2[X, Y](
//    implicit sourceInfo: SourceInfo,
//    compileOptions:      CompileOptions,
//    lookupableX:         Lookupable[X],
//    lookupableY:         Lookupable[Y]
//  ) = new Lookupable[(X, Y)] {
//    type C = (lookupableX.C, lookupableY.C)
//    def definitionLookup[A](that: A => (X, Y), definition: Definition[A]): C = {
//      val ret = that(definition.proto)
//      (
//        lookupableX.definitionLookup[A](_ => ret._1, definition),
//        lookupableY.definitionLookup[A](_ => ret._2, definition)
//      )
//    }
//    def instanceLookup[A](that: A => (X, Y), instance: Instance[A]): C = {
//      import instance._
//      val ret = that(proto)
//      (lookupableX.instanceLookup[A](_ => ret._1, instance), lookupableY.instanceLookup[A](_ => ret._2, instance))
//    }
//  }
}

object Definition {
  def apply[T](proto: => T): Instance[T] =
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