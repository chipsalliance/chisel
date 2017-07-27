// See LICENSE for license details.

// Macro transforms that statically (at compile time) parse range specifiers and emit the raw
// (non-human-friendly) range constructor calls.

package chisel3.internal

import scala.language.experimental.macros
import scala.reflect.macros.blackbox
import scala.util.matching.Regex

/**
  * This is used to parse a Firrtl Interval Range specification.
  * It is complicated by the fact that the string being parsed could be interpolated
  *
  * @param c The Context
  */
class RangeTransform(val c: blackbox.Context) {
  import c.universe._

  //scalastyle:off cyclomatic.complexity
  //scalastyle:off method.length
  def apply(args: c.Tree*): c.Tree = {
    //noinspection ScalaUnusedSymbol
    val stringTrees = c.prefix.tree match {
      case q"$_(scala.StringContext.apply(..$strings))" =>
        strings
      case _ =>
        c.abort(c.enclosingPosition, s"Range macro unable to parse StringContext, got: ${showCode(c.prefix.tree)}")
    }

    val strings = stringTrees.map {
      case Literal(Constant(string: String)) =>
        string
      case tree =>
        c.abort(c.enclosingPosition, s"Range macro unable to parse StringContext element, got: ${showRaw(tree)}")
    }

    var nextStringIndex: Int = 1
    var nextArgIndex: Int = 0
    var currString: String = strings.head

    /** Mutably gets the next value in the range specifier, integer, decimal or question mark(?).
      */
    def getNextValue: c.Tree = {
      currString = currString.dropWhile(_ == ' ') // eat any whitespace

      if (currString.isEmpty) {
        if (nextArgIndex >= args.length) {
          c.abort(c.enclosingPosition, s"Incomplete range specifier")
        }
        val nextArg = args(nextArgIndex)
        nextArgIndex += 1

        if (nextStringIndex >= strings.length) {
          c.abort(c.enclosingPosition, s"Incomplete range specifier")
        }
        currString = strings(nextStringIndex)
        nextStringIndex += 1

        nextArg
      }
      else {
        val nextStringVal = currString match {
          case RangeTransform.DecimalNumber(numberString) => numberString
          case RangeTransform.IntegerNumber(numberString) => numberString
          case RangeTransform.UnspecifiedNumber(_) => "?"
          case _ =>
            c.abort(c.enclosingPosition, s"Bad number or unspecified bound $currString")
        }
        currString = currString.substring(nextStringVal.length)

        if(nextStringVal == "?") {
          Literal(Constant("?"))
        }
        else {
          c.parse(nextStringVal)
        }
      }
    }

    // Currently, not allowed to have the end stops (inclusive / exclusive) be interpolated.
    currString = currString.dropWhile(_ == ' ')
    val startInclusive = currString.headOption match {
      case Some('[') => true
      case Some('(') => false
      case Some(other) => c.abort(c.enclosingPosition, s"Unknown start inclusive/exclusive specifier, got: '$other'")
      case None => c.abort(c.enclosingPosition, s"No initial inclusive/exclusive specifier")
    }

    currString = currString.substring(1) // eat the inclusive/exclusive specifier
    val minArg = getNextValue
    currString = currString.dropWhile(_ == ' ')
    if (currString.isEmpty) {
      c.abort(c.enclosingPosition, s"Incomplete range specifier, missing comma")
    }
    if (currString.head != ',') {
      c.abort(c.enclosingPosition, s"Incomplete range specifier, expected ',', got $currString")
    }

    currString = currString.substring(1) // eat the comma

    val maxArg = getNextValue
    currString = currString.dropWhile(_ == ' ')

    val endInclusive = currString.headOption match {
      case Some(']') => true
      case Some(')') => false
      case Some(other) =>
        c.abort(c.enclosingPosition, s"Unknown end inclusive/exclusive specifier, got: '$other'")
      case None =>
        c.abort(c.enclosingPosition, s"Incomplete range specifier, missing end inclusive/exclusive specifier")
    }
    currString = currString.substring(1) // eat the inclusive/exclusive specifier
    currString = currString.dropWhile(_ == ' ')

    val binaryPointString = currString.headOption match {
      case Some('.') =>
        currString = currString.substring(1)
        getNextValue
      case Some(other) =>
        c.abort(c.enclosingPosition, s"Unknown end inclusive/exclusive specifier, got: '$other'")
      case None =>
        Literal(Constant(0))
    }

    if (nextArgIndex < args.length) {
      val unused = args.mkString("")
      c.abort(c.enclosingPosition, s"Unused interpolated values in range specifier: '$unused'")
    }
    if (!currString.isEmpty || nextStringIndex < strings.length) {
      val unused = currString + strings.slice(nextStringIndex, strings.length).mkString(", ")
      c.abort(c.enclosingPosition, s"Unused characters in range specifier: '$unused'")
    }

    val startBound = q"_root_.chisel3.internal.firrtl.Range.getBound($startInclusive, $minArg)"

    val endBound = q"_root_.chisel3.internal.firrtl.Range.getBound($endInclusive, $maxArg)"

    val binaryPoint = q"_root_.chisel3.internal.firrtl.Range.getBinaryPoint($binaryPointString)"

    q"new Range($startBound, $endBound, $binaryPoint)"
  }
}

object RangeTransform {
  val UnspecifiedNumber: Regex = """(\?).*""".r
  val IntegerNumber: Regex = """(-?\d+).*""".r
  val DecimalNumber: Regex = """(-?\d+\.\d+).*""".r
}
