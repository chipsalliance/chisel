// SPDX-License-Identifier: Apache-2.0

package firrtl.options

import firrtl.AnnotationSeq

import logger.LazyLogging

import scala.collection.mutable.LinkedHashSet

import scala.reflect
import scala.reflect.ClassTag

object Dependency {
  def apply[A <: DependencyAPI[_]: ClassTag]: Dependency[A] = {
    val clazz = reflect.classTag[A].runtimeClass
    Dependency(Left(clazz.asInstanceOf[Class[A]]))
  }

  def apply[A <: DependencyAPI[_]](c: Class[_ <: A]): Dependency[A] = {
    // It's forbidden to wrap the class of a singleton as a Dependency
    require(c.getName.last != '$')
    Dependency(Left(c))
  }

  def apply[A <: DependencyAPI[_]](o: A with Singleton): Dependency[A] = Dependency(Right(o))

  def fromTransform[A <: DependencyAPI[_]](t: A): Dependency[A] = {
    if (isSingleton(t)) {
      Dependency[A](Right(t.asInstanceOf[A with Singleton]))
    } else {
      Dependency[A](Left(t.getClass))
    }
  }

  private def isSingleton(obj: AnyRef): Boolean = {
    reflect.runtime.currentMirror.reflect(obj).symbol.isModuleClass
  }
}

case class Dependency[+A <: DependencyAPI[_]](id: Either[Class[_ <: A], A with Singleton]) {
  def getObject(): A = id match {
    case Left(c)  => safeConstruct(c)
    case Right(o) => o
  }

  def getSimpleName: String = id match {
    case Left(c)  => c.getSimpleName
    case Right(o) => o.getClass.getSimpleName
  }

  def getName: String = id match {
    case Left(c)  => c.getName
    case Right(o) => o.getClass.getName
  }

  /** Wrap an [[IllegalAccessException]] due to attempted object construction in a [[DependencyManagerException]] */
  private def safeConstruct[A](a: Class[_ <: A]): A = try { a.getDeclaredConstructor().newInstance() }
  catch {
    case e: IllegalAccessException =>
      throw new DependencyManagerException(s"Failed to construct '$a'! (Did you try to construct an object?)", e)
    case e: InstantiationException =>
      throw new DependencyManagerException(
        s"Failed to construct '$a'! (Did you try to construct an inner class or a class with parameters?)",
        e
      )
  }
}

/** A polymorphic mathematical transform
  * @tparam A the transformed type
  */
trait TransformLike[A] extends LazyLogging {

  /** An identifier of this [[TransformLike]] that can be used for logging and informational printing */
  def name: String

  /** A mathematical transform on some type
    * @param a an input object
    * @return an output object of the same type
    */
  def transform(a: A): A

}

/** Mix-in that makes a [[firrtl.options.TransformLike TransformLike]] guaranteed to be an identity function on some
  * type.
  * @tparam A the transformed type
  */
trait IdentityLike[A] { this: TransformLike[A] =>

  /** The internal operation of this transform which, in order for this to be an identity function, must return nothing.
    * @param a an input object
    * @return nothing
    */
  protected def internalTransform(a: A): Unit = ()

  /** This method will execute `internalTransform` and then return the original input object
    * @param a an input object
    * @return the input object
    */
  final override def transform(a: A): A = {
    internalTransform(a)
    a
  }

}

/** Mixin that defines dependencies between [[firrtl.options.TransformLike TransformLike]]s (hereafter referred to as
  * "transforms")
  *
  * This trait forms the basis of the Dependency API of the Chisel/FIRRTL Hardware Compiler Framework. Dependencies are
  * defined in terms of prerequisistes, optional prerequisites, optional prerequisites of, and invalidates. A
  * prerequisite is a transform that must run before this transform. An optional prerequisites is transform that should
  * run before this transform if the other transform is a target (or the prerequisite of a target). An optional
  * prerequisite of is an optional prerequisite injected into another transform. Finally, invalidates define the set of
  * transforms whose effects this transform undos/invalidates. (Invalidation then implies that a transform that is
  * invalidated by this transform and needed by another transform will need to be re-run.)
  *
  * This Dependency API only defines dependencies. A concrete [[DependencyManager]] is expected to be used to statically
  * resolve a linear ordering of transforms that satisfies dependency requirements.
  * @tparam A some transform
  * @define seqNote @note The use of a Seq here is to preserve input order. Internally, this will be converted to a private,
  * ordered Set.
  */
trait DependencyAPI[A <: DependencyAPI[A]] { this: TransformLike[_] =>

  /** All transform that must run before this transform
    * $seqNote
    */
  def prerequisites:                        Seq[Dependency[A]] = Seq.empty
  private[options] lazy val _prerequisites: LinkedHashSet[Dependency[A]] = new LinkedHashSet() ++ prerequisites

  /** All transforms that, if a prerequisite of *another* transform, will run before this transform.
    * $seqNote
    */
  def optionalPrerequisites: Seq[Dependency[A]] = Seq.empty
  private[options] lazy val _optionalPrerequisites: LinkedHashSet[Dependency[A]] =
    new LinkedHashSet() ++ optionalPrerequisites

  /** A sequence of transforms to add this transform as an `optionalPrerequisite`. The use of `optionalPrerequisiteOf`
    * enables the transform declaring them to always run before some other transforms. However, declaring
    * `optionalPrerequisiteOf` will not result in the sequence of transforms executing.
    *
    * This is useful for providing an ordering constraint to guarantee that other transforms (e.g., emitters) will not
    * be scheduled before you.
    *
    * @note This method **will not** result in the listed transforms running. If you want to add multiple transforms at
    * once, you should use a `DependencyManager` with multiple targets.
    */
  def optionalPrerequisiteOf: Seq[Dependency[A]] = Seq.empty
  private[options] lazy val _optionalPrerequisiteOf: LinkedHashSet[Dependency[A]] =
    new LinkedHashSet() ++ optionalPrerequisiteOf

  /** A function that, given *another* transform (parameter `a`) will return true if this transform invalidates/undos the
    * effects of the *other* transform (parameter `a`).
    * @param a transform
    */
  def invalidates(a: A): Boolean = true

}

/** A mathematical transformation of an [[AnnotationSeq]].
  *
  * A [[firrtl.options.Phase Phase]] forms one unit in the Chisel/FIRRTL Hardware Compiler Framework (HCF). The HCF is
  * built from a sequence of [[firrtl.options.Phase Phase]]s applied to an [[AnnotationSeq]]. Note that a
  * [[firrtl.options.Phase Phase]] may consist of multiple phases internally.
  */
trait Phase extends TransformLike[AnnotationSeq] with DependencyAPI[Phase] {

  /** The name of this [[firrtl.options.Phase Phase]]. This will be used to generate debug/error messages or when deleting
    * annotations. This will default to the `simpleName` of the class.
    * @return this phase's name
    * @note Override this with your own implementation for different naming behavior.
    */
  lazy val name: String = this.getClass.getName

}

/** A [[firrtl.options.TransformLike TransformLike]] that internally ''translates'' the input type to some other type,
  * transforms the internal type, and converts back to the original type.
  *
  * This is intended to be used to insert a [[firrtl.options.TransformLike TransformLike]] parameterized by type `B`
  * into a sequence of [[firrtl.options.TransformLike TransformLike]]s parameterized by type `A`.
  * @tparam A the type of the [[firrtl.options.TransformLike TransformLike]]
  * @tparam B the internal type
  */
trait Translator[A, B] extends TransformLike[A] {

  /** A method converting type `A` into type `B`
    * @param an object of type `A`
    * @return an object of type `B`
    */
  protected implicit def aToB(a: A): B

  /** A method converting type `B` back into type `A`
    * @param an object of type `B`
    * @return an object of type `A`
    */
  protected implicit def bToA(b: B): A

  /** A transform on an internal type
    * @param b an object of type `B`
    * @return an object of type `B`
    */
  protected def internalTransform(b: B): B

  /** Convert the input object to the internal type, transform the internal type, and convert back to the original type
    */
  override final def transform(a: A): A = bToA(internalTransform(aToB(a)))

}
