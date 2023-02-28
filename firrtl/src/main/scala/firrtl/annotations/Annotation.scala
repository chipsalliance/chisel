// SPDX-License-Identifier: Apache-2.0

package firrtl
package annotations

import org.json4s.JValue

case class AnnotationException(message: String) extends Exception(message)

/** Base type of auxiliary information */
trait Annotation extends Product {

  /** Update the target based on how signals are renamed */
  def update(renames: RenameMap): Seq[Annotation]

  /** Optional pretty print
    *
    * @note rarely used
    */
  def serialize: String = this.toString

}

/** If an Annotation does not target any [[Named]] thing in the circuit, then all updates just
  * return the Annotation itself
  */
trait NoTargetAnnotation extends Annotation {
  def update(renames: RenameMap): Seq[NoTargetAnnotation] = Seq(this)
}

/** An Annotation that targets a single [[Named]] thing */
trait SingleTargetAnnotation[T <: Named] extends Annotation {
  val target: T

  /** Create another instance of this Annotation */
  def duplicate(n: T): Annotation

  // This mess of @unchecked and try-catch is working around the fact that T is unknown due to type
  // erasure. We cannot that newTarget is of type T, but a CastClassException will be thrown upon
  // invoking duplicate if newTarget cannot be cast to T (only possible in the concrete subclass)
  def update(renames: RenameMap): Seq[Annotation] = {
    target match {
      case c: Target =>
        val x = renames.get(c)
        x.map(newTargets => newTargets.map(t => duplicate(t.asInstanceOf[T]))).getOrElse(List(this))
      case from: Named =>
        val ret = renames.get(Target.convertNamed2Target(target))
        ret
          .map(_.map { newT =>
            val result = newT match {
              case c: InstanceTarget => ModuleName(c.ofModule, CircuitName(c.circuit))
              case c: IsMember =>
                val local = Target.referringModule(c)
                c.setPathTarget(local)
              case c: CircuitTarget => c.toNamed
              case other => throw Target.NamedException(s"Cannot convert $other to [[Named]]")
            }
            (Target.convertTarget2Named(result): @unchecked) match {
              case newTarget: T @unchecked =>
                try {
                  duplicate(newTarget)
                } catch {
                  case _: java.lang.ClassCastException =>
                    val msg = s"${this.getClass.getName} target ${target.getClass.getName} " +
                      s"cannot be renamed to ${newTarget.getClass}"
                    throw AnnotationException(msg)
                }
            }
          })
          .getOrElse(List(this))
    }
  }
}

object Annotation

case class UnrecognizedAnnotation(underlying: JValue) extends NoTargetAnnotation
