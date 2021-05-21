// SPDX-License-Identifier: Apache-2.0

package firrtl
package annotations

import firrtl.options.StageUtils

import scala.collection.Traversable

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

  /** Recurses through ls to find all [[Target]] instances
    * @param ls
    * @return
    */
  private def extractComponents(ls: Traversable[_]): Traversable[Target] = {
    ls.flatMap {
      case c: Target                          => Seq(c)
      case x: scala.collection.Traversable[_] => extractComponents(x)
      case o: Product                         => extractComponents(o.productIterator.toIterable)
      case _ => Seq()
    }
  }

  /** Returns all [[firrtl.annotations.Target Target]] members in this annotation
    * @return
    */
  def getTargets: Seq[Target] = extractComponents(productIterator.toIterable).toSeq
}

/** If an Annotation does not target any [[Named]] thing in the circuit, then all updates just
  * return the Annotation itself
  */
trait NoTargetAnnotation extends Annotation {
  def update(renames: RenameMap): Seq[NoTargetAnnotation] = Seq(this)

  override def getTargets: Seq[Target] = Seq.empty
}

/** An Annotation that targets a single [[Named]] thing */
trait SingleTargetAnnotation[T <: Named] extends Annotation {
  val target: T

  // we can implement getTargets more efficiently since we know that we have exactly one target
  override def getTargets: Seq[Target] = Seq(target)

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

/** [[MultiTargetAnnotation]] keeps the renamed targets grouped within a single annotation. */
trait MultiTargetAnnotation extends Annotation {

  /** Contains a nested sequence of [[firrtl.annotations.Target Target]]
    *
    * Each inner Seq should contain a single element. For example:
    * {{{
    * def targets = Seq(Seq(foo), Seq(bar))
    * }}}
    */
  def targets: Seq[Seq[Target]]

  override def getTargets: Seq[Target] = targets.flatten

  /** Create another instance of this Annotation
    *
    * The inner Seqs correspond to the renames of the inner Seqs of targets
    */
  def duplicate(n: Seq[Seq[Target]]): Annotation

  /** Assume [[RenameMap]] is `Map(TargetA -> Seq(TargetA1, TargetA2, TargetA3), TargetB -> Seq(TargetB1, TargetB2))`
    * in the update, this Annotation is still one annotation, but the contents are renamed in the below form
    * Seq(Seq(TargetA1, TargetA2, TargetA3), Seq(TargetB1, TargetB2), Seq(TargetC))
    */
  def update(renames: RenameMap): Seq[Annotation] = Seq(duplicate(targets.map(ts => ts.flatMap(renames(_)))))

  private def crossJoin[T](list: Seq[Seq[T]]): Seq[Seq[T]] =
    list match {
      case Nil      => Nil
      case x :: Nil => x.map(Seq(_))
      case x :: xs =>
        val xsJoin = crossJoin(xs)
        for {
          i <- x
          j <- xsJoin
        } yield {
          Seq(i) ++ j
        }
    }

  /** Assume [[RenameMap]] is `Map(TargetA -> Seq(TargetA1, TargetA2, TargetA3), TargetB -> Seq(TargetB1, TargetB2))`
    * After flat, this Annotation will be flat to the [[AnnotationSeq]] in the below form
    * Seq(Seq(TargetA1), Seq(TargetB1), Seq(TargetC)); Seq(Seq(TargetA1), Seq(TargetB2), Seq(TargetC))
    * Seq(Seq(TargetA2), Seq(TargetB1), Seq(TargetC)); Seq(Seq(TargetA2), Seq(TargetB2), Seq(TargetC))
    * Seq(Seq(TargetA3), Seq(TargetB1), Seq(TargetC)); Seq(Seq(TargetA3), Seq(TargetB2), Seq(TargetC))
    */
  def flat(): AnnotationSeq = crossJoin(targets).map(r => duplicate(r.map(Seq(_))))
}

object Annotation

case class DeletedAnnotation(xFormName: String, anno: Annotation) extends NoTargetAnnotation {
  override def serialize: String = s"""DELETED by $xFormName\n${anno.serialize}"""
}
