// See LICENSE for license details.

// Macro transforms that statically (at compile time) parse range specifiers and emit the raw
// (non-human-friendly) range constructor calls.

package chisel3.internal

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context
import scala.reflect.macros.whitebox

// Workaround for https://github.com/sbt/sbt/issues/3966
object RangeTransform
class RangeTransform(val c: Context) {
  import c.universe._
  def apply(args: c.Tree*): c.Tree = {
    val stringTrees = c.prefix.tree match {
      case q"$_(scala.StringContext.apply(..$strings))" => strings
      case _ => c.abort(c.enclosingPosition, s"Range macro unable to parse StringContext, got: ${showCode(c.prefix.tree)}")
    }
    val strings = stringTrees.map { tree => tree match {
      case Literal(Constant(string: String)) => string
      case _ => c.abort(c.enclosingPosition, s"Range macro unable to parse StringContext element, got: ${showRaw(tree)}")
    } }

    var nextStringIndex: Int = 1
    var nextArgIndex: Int = 0
    var currString: String = strings(0)

    /** Mutably gets the next numeric value in the range specifier.
      */
    def getNextValue(): c.Tree = {
      currString = currString.dropWhile(_ == ' ')  // allow whitespace
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
      } else {
        val nextStringVal = currString.takeWhile(!Set('[', '(', ' ', ',', ')', ']').contains(_))
        currString = currString.substring(nextStringVal.length)
        if (currString.isEmpty) {
          c.abort(c.enclosingPosition, s"Incomplete range specifier")
        }
        c.parse(nextStringVal)
      }
    }

    // Currently, not allowed to have the end stops (inclusive / exclusive) be interpolated.
    currString = currString.dropWhile(_ == ' ')
    val startInclusive = currString(0) match {
      case '[' => true
      case '(' => false
      case other => c.abort(c.enclosingPosition, s"Unknown start inclusive/exclusive specifier, got: '$other'")
    }
    currString = currString.substring(1)  // eat the inclusive/exclusive specifier
    val minArg = getNextValue()
    currString = currString.dropWhile(_ == ' ')
    if (currString(0) != ',') {
      c.abort(c.enclosingPosition, s"Incomplete range specifier, expected ','")
    }
    currString = currString.substring(1)  // eat the comma
    val maxArg = getNextValue()
    currString = currString.dropWhile(_ == ' ')
    val endInclusive = currString(0) match {
      case ']' => true
      case ')' => false
      case other => c.abort(c.enclosingPosition, s"Unknown end inclusive/exclusive specifier, got: '$other'")
    }
    currString = currString.substring(1)  // eat the inclusive/exclusive specifier
    currString = currString.dropWhile(_ == ' ')

    if (nextArgIndex < args.length) {
      val unused = args.mkString("")
      c.abort(c.enclosingPosition, s"Unused interpolated values in range specifier: '$unused'")
    }
    if (!currString.isEmpty || nextStringIndex < strings.length) {
      val unused = currString + strings.slice(nextStringIndex, strings.length).mkString(", ")
      c.abort(c.enclosingPosition, s"Unused characters in range specifier: '$unused'")
    }

    val startBound = if (startInclusive) {
      q"_root_.chisel3.internal.firrtl.Closed($minArg)"
    } else {
      q"_root_.chisel3.internal.firrtl.Open($minArg)"
    }
    val endBound = if (endInclusive) {
      q"_root_.chisel3.internal.firrtl.Closed($maxArg)"
    } else {
      q"_root_.chisel3.internal.firrtl.Open($maxArg)"
    }

    q"($startBound, $endBound)"
  }
}
