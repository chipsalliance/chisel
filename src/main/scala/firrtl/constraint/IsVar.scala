// See LICENSE for license details.

package firrtl.constraint

object IsVar {
  def unapply(i: Constraint): Option[String] = i match {
    case i: IsVar => Some(i.name)
    case _ => None
  }
}

/** Extend to be a constraint variable */
trait IsVar extends Constraint {

  def name: String

  override def serialize: String = name

  override def map(f: Constraint=>Constraint): Constraint = this

  override def reduce() = this

  val children = Vector()
}

case class VarCon(name: String) extends IsVar

