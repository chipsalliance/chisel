// See LICENSE for license details.

package firrtl

import scala.collection.mutable
import scala.collection.mutable.HashSet
import firrtl.ir._
import Mappers._

class Namespace private {
  private val tempNamePrefix: String = "_GEN"
  // Begin with a tempNamePrefix in namespace so we always have a number suffix
  private val namespace = mutable.HashSet[String](tempNamePrefix)
  private var tempN = 0

  def tryName(value: String): Boolean = {
    val unused = !contains(value)
    if (unused) namespace += value
    unused
  }

  def contains(value: String): Boolean = namespace.contains(value)

  private def newNameIndex(value: String, idx: Int): (String, Int) = {
    var n = idx
    var str = value
    while (!tryName(str)) {
      str = s"${value}_$n"
      n += 1
    }
    (str, n)
  }

  def newName(value: String): String = newNameIndex(value, 0)._1

  def newTemp: String = {
    val (name, n) = newNameIndex(tempNamePrefix, tempN)
    tempN = n
    name
  }
}

/* TODO(azidar): Make Namespace return unique names that will not conflict with expanded
 * names after LowerTypes expands names (like the Uniquify pass).
 */
object Namespace {
  // Initializes a namespace from a Module
  def apply(m: DefModule): Namespace = {
    val namespace = new Namespace

    def buildNamespaceStmt(s: Statement): Seq[String] = s match {
      case s: IsDeclaration => Seq(s.name)
      case s: Conditionally => buildNamespaceStmt(s.conseq) ++ buildNamespaceStmt(s.alt)
      case s: Block => s.stmts flatMap buildNamespaceStmt
      case _ => Nil
    }
    namespace.namespace ++= m.ports map (_.name)
    m match {
      case in: Module =>
        namespace.namespace ++= buildNamespaceStmt(in.body)
      case _ => // Do nothing
    }

    namespace
  }

  /** Initializes a [[Namespace]] for [[ir.Module]] names in a [[ir.Circuit]] */
  def apply(c: Circuit): Namespace = {
    val namespace = new Namespace
    namespace.namespace ++= c.modules map (_.name)
    namespace
  }

  /** Initializes a [[Namespace]] from arbitrary strings **/
  def apply(names: Seq[String] = Nil): Namespace = {
    val namespace = new Namespace
    namespace.namespace ++= names
    namespace
  }
}

