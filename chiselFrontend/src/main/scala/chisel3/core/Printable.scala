// See LICENSE for license details.

package chisel3.core

import chisel3.internal.firrtl.Component
import chisel3.internal.HasId

import scala.collection.mutable

import java.util.{
  MissingFormatArgumentException,
  UnknownFormatConversionException
}

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
  *   printf(p"The value of wire = %d\n", wire)
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
  final def +(that: Printable) = Printables(List(this, that))
  /** Allow for appending Strings to Printables */
  final def +(that: String) = Printables(List(this, PString(that)))
}
object Printable {
  /** Pack standard printf fmt, args* style into Printable
    */
  def pack(fmt: String, data: Data*): Printable = {
    val args = data.toIterator

    // Error handling
    def carrotAt(index: Int) = (" " * index) + "^"
    def errorMsg(index: Int) =
      s"""|    fmt = "$fmt"
          |           ${carrotAt(index)}
          |    data = ${data mkString ", "}""".stripMargin
    def getArg(i: Int): Data = {
      if (!args.hasNext) {
        val msg = "has no matching argument!\n" + errorMsg(i)
          // Exception wraps msg in s"Format Specifier '$msg'"
        throw new MissingFormatArgumentException(msg)
      }
      args.next()
    }

    val pables = mutable.ListBuffer.empty[Printable]
    var str = ""
    var percent = false
    for ((c, i) <- fmt.zipWithIndex) {
      if (percent) {
        val arg = c match {
          case FirrtlFormat(x) => FirrtlFormat(x.toString, getArg(i))
          case 'n' => Name(getArg(i))
          case 'N' => FullName(getArg(i))
          case '%' => Percent
          case x =>
            val msg = s"Illegal format specifier '$x'!\n" + errorMsg(i)
            throw new UnknownFormatConversionException(msg)
        }
        pables += PString(str dropRight 1) // remove format %
        pables += arg
        str = ""
        percent = false
      } else {
        str += c
        percent = c == '%'
      }
    }
    if (percent) {
      val msg = s"Trailing %\n" + errorMsg(fmt.size - 1)
      throw new UnknownFormatConversionException(msg)
    }
    require(!args.hasNext,
      s"Too many arguments! More format specifier(s) expected!\n" +
      errorMsg(fmt.size))

    pables += PString(str)
    Printables(pables)
  }
}

case class Printables(pables: Iterable[Printable]) extends Printable {
  require(pables.hasDefiniteSize, "Infinite-sized iterables are not supported!")
  final def unpack(ctx: Component): (String, Iterable[String]) = {
    val (fmts, args) = pables.map(_ unpack ctx).unzip
    (fmts.mkString, args.flatten)
  }
}
/** Wrapper for printing Scala Strings */
case class PString(str: String) extends Printable {
  final def unpack(ctx: Component): (String, Iterable[String]) =
    (str replaceAll ("%", "%%"), List.empty)
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
    Option(x) filter (x => legalSpecifiers contains x)

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
      case c => throw new Exception(s"Illegal format specifier '$c'!")
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
