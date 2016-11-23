// See LICENSE for license details.

package firrtl
package annotations

import firrtl.ir._

object AnnotationUtils {
  /** Returns true if a valid Module name */
  val SerializedModuleName = """([a-zA-Z_][a-zA-Z_0-9~!@#$%^*\-+=?/]*)""".r
  def validModuleName(s: String): Boolean = s match {
    case SerializedModuleName(name) => true
    case _ => false
  }

  /** Returns true if a valid component/subcomponent name */
  val SerializedComponentName = """([a-zA-Z_][a-zA-Z_0-9\[\]\.~!@#$%^*\-+=?/]*)""".r
  def validComponentName(s: String): Boolean = s match {
    case SerializedComponentName(name) => true
    case _ => false
  }

  /** Tokenizes a string with '[', ']', '.' as tokens, e.g.:
   *  "foo.bar[boo.far]" becomes Seq("foo" "." "bar" "[" "boo" "." "far" "]")
   */
  def tokenize(s: String): Seq[String] = s.find(c => "[].".contains(c)) match {
    case Some(_) =>
      val i = s.indexWhere(c => "[].".contains(c))
      s.slice(0, i) match {
        case "" => Seq(s(i).toString) ++ tokenize(s.drop(i + 1))
        case x => Seq(x, s(i).toString) ++ tokenize(s.drop(i + 1))
      }
    case None if s == "" => Nil
    case None => Seq(s)
  }

  /** Given a serialized component/subcomponent reference, subindex, subaccess,
   *  or subfield, return the corresponding IR expression.
   *  E.g. "foo.bar" becomes SubField(Reference("foo", UnknownType), "bar", UnknownType)
   */
  def toExp(s: String): Expression = {
    def parse(tokens: Seq[String]): Expression = {
      val DecPattern = """([1-9]\d*)""".r
      def findClose(tokens: Seq[String], index: Int, nOpen: Int): Seq[String] = {
        if(index >= tokens.size) {
          error("Cannot find closing bracket ]")
        } else tokens(index) match {
          case "[" => findClose(tokens, index + 1, nOpen + 1)
          case "]" if nOpen == 1 => tokens.slice(1, index)
          case "]" => findClose(tokens, index + 1, nOpen - 1)
          case _ => findClose(tokens, index + 1, nOpen)
        }
      }
      def buildup(e: Expression, tokens: Seq[String]): Expression = tokens match {
        case "[" :: tail =>
          val indexOrAccess = findClose(tokens, 0, 0)
          val exp = indexOrAccess.head match {
            case DecPattern(d) => SubIndex(e, d.toInt, UnknownType)
            case _ => SubAccess(e, parse(indexOrAccess), UnknownType)
          }
          buildup(exp, tokens.drop(2 + indexOrAccess.size))
        case "." :: tail =>
          buildup(SubField(e, tokens(1), UnknownType), tokens.drop(2))
        case Nil => e
      }
      val root = Reference(tokens.head, UnknownType)
      buildup(root, tokens.tail)
    }
    if(validComponentName(s)) {
      parse(tokenize(s))
    } else error(s"Cannot convert $s into an expression.")
  }
}

