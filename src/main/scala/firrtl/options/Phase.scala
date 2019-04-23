// See LICENSE for license details.

package firrtl.options

import firrtl.AnnotationSeq
import firrtl.annotations.DeletedAnnotation

import logger.LazyLogging

import scala.collection.mutable

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

/** A mathematical transformation of an [[AnnotationSeq]].
  *
  * A [[Phase]] forms one unit in the Chisel/FIRRTL Hardware Compiler Framework (HCF). The HCF is built from a sequence
  * of [[Phase]]s applied to an [[AnnotationSeq]]. Note that a [[Phase]] may consist of multiple phases internally.
  */
abstract class Phase extends TransformLike[AnnotationSeq] {

  /** The name of this [[Phase]]. This will be used to generate debug/error messages or when deleting annotations. This
    * will default to the `simpleName` of the class.
    * @return this phase's name
    * @note Override this with your own implementation for different naming behavior.
    */
  lazy val name: String = this.getClass.getName

}

/** A [[TransformLike]] that internally ''translates'' the input type to some other type, transforms the internal type,
  * and converts back to the original type.
  *
  * This is intended to be used to insert a [[TransformLike]] parameterized by type `B` into a sequence of
  * [[TransformLike]]s parameterized by type `A`.
  * @tparam A the type of the [[TransformLike]]
  * @tparam B the internal type
  */
trait Translator[A, B] { this: TransformLike[A] =>

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
  final def transform(a: A): A = internalTransform(a)

}
