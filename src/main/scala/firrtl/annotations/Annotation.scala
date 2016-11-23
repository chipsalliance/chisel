// See LICENSE for license details.

package firrtl
package annotations

import firrtl.ir._

case class AnnotationException(message: String) extends Exception(message)

final case class Annotation(target: Named, transform: Class[_ <: Transform], value: String) {
  val targetString: String = target.serialize
  val transformClass: String = transform.getName
  def serialize: String = this.toString
  def update(tos: Seq[Named]): Seq[Annotation] = {
    check(target, tos, this)
    propagate(target, tos, duplicate)
  }
  def propagate(from: Named, tos: Seq[Named], dup: Named=>Annotation): Seq[Annotation] = tos.map(dup(_))
  def check(from: Named, tos: Seq[Named], which: Annotation): Unit = {}
  def duplicate(n: Named) = new Annotation(n, transform, value)
}
