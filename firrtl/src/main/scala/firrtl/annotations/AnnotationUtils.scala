// SPDX-License-Identifier: Apache-2.0

package firrtl
package annotations

import java.io.File

import firrtl.ir._

case class InvalidAnnotationFileException(file: File, cause: FirrtlUserException = null)
    extends FirrtlUserException(s"$file", cause)
case class UnrecogizedAnnotationsException(msg: String) extends FirrtlUserException(s"Unrecognized annotations $msg")
case class InvalidAnnotationJSONException(msg: String) extends FirrtlUserException(msg)
case class AnnotationFileNotFoundException(file: File)
    extends FirrtlUserException(
      s"Annotation file $file not found!"
    )
case class AnnotationClassNotFoundException(className: String)
    extends FirrtlUserException(
      s"Annotation class $className not found! Please check spelling and classpath"
    )
class UnserializableAnnotationException private (msg: String) extends FirrtlUserException(msg)
object UnserializableAnnotationException {
  private def toMessage(pair: (Annotation, Throwable)): String =
    s"Failed to serialiaze annotation of type ${pair._1.getClass.getName} because '${pair._2.getMessage}'"
  private[firrtl] def apply(badAnnos: Seq[(Annotation, Throwable)]) = {
    require(badAnnos.nonEmpty)
    val msg = badAnnos.map(toMessage).mkString("\n  ", "\n  ", "\n")
    new UnserializableAnnotationException(msg)
  }
}

object AnnotationUtils {

  /** Returns true if a valid Module name */
  val SerializedModuleName = """([a-zA-Z_][a-zA-Z_0-9~!@#$%^*\-+=?/]*)""".r
  def validModuleName(s: String): Boolean = s match {
    case SerializedModuleName(name) => true
    case _                          => false
  }

  /** Returns true if a valid component/subcomponent name */
  val SerializedComponentName = """([a-zA-Z_][a-zA-Z_0-9\[\]\.~!@#$%^*\-+=?/]*)""".r
  def validComponentName(s: String): Boolean = s match {
    case SerializedComponentName(name) => true
    case _                             => false
  }

  /** Tokenizes a string with '[', ']', '.' as tokens, e.g.:
    *  "foo.bar[boo.far]" becomes Seq("foo" "." "bar" "[" "boo" "." "far" "]")
    */
  def tokenize(s: String): Seq[String] = s.find(c => "[].".contains(c)) match {
    case Some(_) =>
      val i = s.indexWhere(c => "[].".contains(c))
      s.slice(0, i) match {
        case "" => s(i).toString +: tokenize(s.drop(i + 1))
        case x  => x +: s(i).toString +: tokenize(s.drop(i + 1))
      }
    case None if s == "" => Nil
    case None            => Seq(s)
  }

  def toNamed(s: String): Named = s.split("\\.", 3) match {
    case Array(n)       => CircuitName(n)
    case Array(c, m)    => ModuleName(m, CircuitName(c))
    case Array(c, m, x) => ComponentName(x, ModuleName(m, CircuitName(c)))
  }

  /** Converts a serialized FIRRTL component into a sequence of target tokens
    * @param s
    * @return
    */
  def toSubComponents(s: String): Seq[TargetToken] = {
    import TargetToken._
    def exp2subcomp(e: ir.Expression): Seq[TargetToken] = e match {
      case ir.Reference(name, _)      => Seq(Ref(name))
      case ir.SubField(expr, name, _) => exp2subcomp(expr) :+ Field(name)
      case ir.SubIndex(expr, idx, _)  => exp2subcomp(expr) :+ Index(idx)
      case ir.SubAccess(expr, idx, _) =>
        Utils.throwInternalError(s"For string $s, cannot convert a subaccess $e into a Target")
    }
    exp2subcomp(toExp(s))
  }

  /** Given a serialized component/subcomponent reference, subindex, subaccess,
    *  or subfield, return the corresponding IR expression.
    *  E.g. "foo.bar" becomes SubField(Reference("foo", UnknownType), "bar", UnknownType)
    */
  def toExp(s: String): Expression = {
    def parse(tokens: Seq[String]): Expression = {
      val DecPattern = """(\d+)""".r
      def findClose(tokens: Seq[String], index: Int, nOpen: Int): Seq[String] = {
        if (index >= tokens.size) {
          Utils.error("Cannot find closing bracket ]")
        } else
          tokens(index) match {
            case "["               => findClose(tokens, index + 1, nOpen + 1)
            case "]" if nOpen == 1 => tokens.slice(1, index)
            case "]"               => findClose(tokens, index + 1, nOpen - 1)
            case _                 => findClose(tokens, index + 1, nOpen)
          }
      }
      def buildup(e: Expression, tokens: Seq[String]): Expression = tokens match {
        case "[" :: tail =>
          val indexOrAccess = findClose(tokens, 0, 0)
          val exp = indexOrAccess.head match {
            case DecPattern(d) => SubIndex(e, d.toInt, UnknownType)
            case _             => SubAccess(e, parse(indexOrAccess), UnknownType)
          }
          buildup(exp, tokens.drop(2 + indexOrAccess.size))
        case "." :: tail =>
          buildup(SubField(e, tokens(1), UnknownType), tokens.drop(2))
        case Nil => e
      }
      val root = Reference(tokens.head, UnknownType)
      buildup(root, tokens.tail)
    }
    if (validComponentName(s)) {
      parse(tokenize(s))
    } else {
      Utils.error(s"Cannot convert $s into an expression.")
    }
  }
}
