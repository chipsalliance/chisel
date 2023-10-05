// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal.firrtl.Component

import scala.collection.mutable

import java.util.{MissingFormatArgumentException, UnknownFormatConversionException}

/** Superclass of things that can be printed in the resulting circuit
  *
  * Usually created using the custom string interpolator `p"..."`. Printable string interpolation is
  * similar to [[https://docs.scala-lang.org/overviews/core/string-interpolation.html String
  * interpolation in Scala]] For example:
  * {{{
  *   printf(p"The value of wire = \$wire\n")
  * }}}
  * This is equivalent to writing:
  * {{{
  *   printf("The value of wire = %d\n", wire)
  * }}}
  * All Chisel data types have a method `.toPrintable` that gives a default pretty print that can be
  * accessed via `p"..."`. This works even for aggregate types, for example:
  * {{{
  *   val myVec = VecInit(5.U, 10.U, 13.U)
  *   printf(p"myVec = \$myVec\n")
  *   // myVec = Vec(5, 10, 13)
  *
  *   val myBundle = Wire(new Bundle {
  *     val foo = UInt()
  *     val bar = UInt()
  *   })
  *   myBundle.foo := 3.U
  *   myBundle.bar := 11.U
  *   printf(p"myBundle = \$myBundle\n")
  *   // myBundle = Bundle(a -> 3, b -> 11)
  * }}}
  * Users can override the default behavior of `.toPrintable` in custom [[Bundle]] and [[Record]]
  * types.
  */
// TODO Add support for names of Modules
//   Currently impossible because unpack is called before the name is selected
//   Could be implemented by adding a new format specifier to Firrtl (eg. %m)
// TODO Should we provide more functions like map and mkPrintable?
sealed abstract class Printable {

  /** Unpack into format String and a List of String arguments (identifiers)
    * @note This must be called after elaboration when Chisel nodes actually
    *   have names
    */
  def unpack(ctx: Component): (String, Iterable[String])

  /** Allow for appending Printables like Strings */
  final def +(that: Printable): Printables = Printables(List(this, that))

  /** Allow for appending Strings to Printables */
  final def +(that: String): Printables = Printables(List(this, PString(that)))
}
object Printable {

  /** Pack standard printf fmt, args* style into Printable
    */
  def pack(fmt: String, data: Data*): Printable = {
    val args = data.iterator
    // Error handling
    def carrotAt(index: Int) = (" " * index) + "^"
    def errorMsg(index: Int) =
      s"""|    fmt = "$fmt"
          |           ${carrotAt(index)}
          |    data = ${data.mkString(", ")}""".stripMargin

    def checkArg(i: Int): Unit = {
      if (!args.hasNext) {
        val msg = "has no matching argument!\n" + errorMsg(i)
        // Exception wraps msg in s"Format Specifier '$msg'"
        throw new MissingFormatArgumentException(msg)
      }
      val _ = args.next()
    }
    var iter = 0
    var curr_start = 0
    val buf = mutable.ListBuffer.empty[String]
    while (iter < fmt.size) {
      // Encountered % which is either
      // 1. Describing a format specifier.
      // 2. Literal Percent
      // 3. Dangling percent - most likely due to a typo - intended literal percent or forgot the specifier.
      // Try to give meaningful error reports
      if (fmt(iter) == '%') {
        if (iter != fmt.size - 1 && (fmt(iter + 1) != '%' && !fmt(iter + 1).isWhitespace)) {
          checkArg(iter)
          buf += fmt.substring(curr_start, iter)
          curr_start = iter
          iter += 1
        }

        // Last character is %.
        else if (iter == fmt.size - 1) {
          val msg = s"Trailing %\n" + errorMsg(fmt.size - 1)
          throw new UnknownFormatConversionException(msg)
        }

        // A lone %
        else if (fmt(iter + 1).isWhitespace) {
          val msg = s"Unescaped % - add % if literal or add proper specifier if not\n" + errorMsg(iter + 1)
          throw new UnknownFormatConversionException(msg)
        }

        // A literal percent - hence increment by 2.
        else {
          iter += 2
        }
      }

      // Normal progression
      else {
        iter += 1
      }
    }
    require(
      !args.hasNext,
      s"Too many arguments! More format specifier(s) expected!\n" +
        errorMsg(fmt.size)
    )
    buf += fmt.substring(curr_start, iter)

    // The string received as an input to pack is already
    // treated  i.e. escape sequences are processed.
    // Since StringContext API assumes the parts are un-treated
    // treatEscapes is called within the implemented custom interpolators.
    // The literal \ needs to be escaped before sending to the custom cf interpolator.

    val bufEscapeBackSlash = buf.map(_.replace("\\", "\\\\"))
    StringContext(bufEscapeBackSlash.toSeq: _*).cf(data: _*)
  }

  private[chisel3] def checkScope(message: Printable): Unit = {
    def getData(x: Printable): Seq[Data] = {
      x match {
        case y: FirrtlFormat => Seq(y.bits)
        case Name(d)       => Seq(d)
        case FullName(d)   => Seq(d)
        case Printables(p) => p.flatMap(getData(_)).toSeq
        case _             => Seq() // Handles subtypes PString and Percent
      }
    }
    getData(message).foreach(_.requireVisible())
  }
}

case class Printables(pables: Iterable[Printable]) extends Printable {
  require(pables.hasDefiniteSize, "Infinite-sized iterables are not supported!")
  final def unpack(ctx: Component): (String, Iterable[String]) = {
    val (fmts, args) = pables.map(_.unpack(ctx)).unzip
    (fmts.mkString, args.flatten)
  }
}

/** Wrapper for printing Scala Strings */
case class PString(str: String) extends Printable {
  final def unpack(ctx: Component): (String, Iterable[String]) =
    (str.replaceAll("%", "%%"), List.empty)
}

/** Superclass for Firrtl format specifiers for Bits */
sealed abstract class FirrtlFormat(private[chisel3] val specifier: Char) extends Printable {
  def bits: Bits
  def unpack(ctx: Component): (String, Iterable[String]) = {
    (s"%$specifier", List(bits.ref.fullName(ctx)))
  }
}
object FirrtlFormat {
  final val legalSpecifiers = List('d', 'x', 'b', 'c')

  def unapply(x: Char): Option[Char] =
    Option(x).filter(x => legalSpecifiers contains x)

  /** Helper for constructing Firrtl Formats
    * Accepts data to simplify pack
    */
  def apply(specifier: String, data: Data): FirrtlFormat = {
    val bits = data match {
      case b: Bits => b
      case d => throw new Exception(s"Trying to construct FirrtlFormat with non-bits $d!")
    }
    specifier match {
      case "d" => Decimal(bits)
      case "x" => Hexadecimal(bits)
      case "b" => Binary(bits)
      case "c" => Character(bits)
      case c   => throw new Exception(s"Illegal format specifier '$c'!")
    }
  }
}

/** Format bits as Decimal */
case class Decimal(bits: Bits) extends FirrtlFormat('d')

/** Format bits as Hexidecimal */
case class Hexadecimal(bits: Bits) extends FirrtlFormat('x')

/** Format bits as Binary */
case class Binary(bits: Bits) extends FirrtlFormat('b')

/** Format bits as Character */
case class Character(bits: Bits) extends FirrtlFormat('c')

/** Put innermost name (eg. field of bundle) */
case class Name(data: Data) extends Printable {
  final def unpack(ctx: Component): (String, Iterable[String]) = (data.ref.name, List.empty)
}

/** Put full name within parent namespace (eg. bundleName.field) */
case class FullName(data: Data) extends Printable {
  final def unpack(ctx: Component): (String, Iterable[String]) = (data.ref.fullName(ctx), List.empty)
}

/** Represents escaped percents */
case object Percent extends Printable {
  final def unpack(ctx: Component): (String, Iterable[String]) = ("%%", List.empty)
}
