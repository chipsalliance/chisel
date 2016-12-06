// See LICENSE for license details.

package firrtl
package annotations

case class AnnotationException(message: String) extends Exception(message)

final case class Annotation(target: Named, transform: Class[_ <: Transform], value: String) {
  val targetString: String = target.serialize
  val transformClass: String = transform.getName

  /**
    * This serialize is basically a pretty printer, actual serialization is handled by
    * AnnotationYamlProtocol
    * @return a nicer string than the raw case class default
    */
  def serialize: String = {
    s"Annotation(${target.serialize},${transform.getCanonicalName},$value)"
  }

  def update(tos: Seq[Named]): Seq[Annotation] = {
    check(target, tos, this)
    propagate(target, tos, duplicate)
  }
  def propagate(from: Named, tos: Seq[Named], dup: Named=>Annotation): Seq[Annotation] = tos.map(dup(_))
  def check(from: Named, tos: Seq[Named], which: Annotation): Unit = {}
  def duplicate(n: Named) = Annotation(n, transform, value)
}
