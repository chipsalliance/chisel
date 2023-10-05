// SPDX-License-Identifier: Apache-2.0

import scala.collection.mutable.ArrayBuffer

/** This is a simplified version of https://github.com/milessabin/shapeless/blob/80d26bc132139eb887fe1e8d39abbdb8dcb20117/project/Boilerplate.scala
  *
  * Generate a range of boilerplate classes, those offering alternatives with 0-22 params
  * and would be tedious to craft by hand
  */
object Boilerplate {

  sealed trait Arg

  /** Interpolate the current arity
    */
  case object Arity extends Arg

  /** Interpolate a string literal verbatim
    */
  case class Literal(str: String) extends Arg

  /** Interpolate a string using a generator function that is called for the range [1, $Arity]
    *
    * i.e. interpolate the string (1 to $Arity).map(fn).mkString(start = start, sep = sep, end = end)
    */
  case class Repeated(
    fn:    Int => String,
    start: String = "",
    sep:   String = ", ",
    end:   String = "")
      extends Arg

  private class Line(start: String) {
    private val stripped = start.dropWhile(_.isWhitespace)
    private val parts = ArrayBuffer[Arg](Literal(stripped.tail))

    val isRepeated = stripped.head match {
      case '-' => true
      case '|' => false
      case _   => throw new IllegalArgumentException(s"block lines must begin '|' or '-': $start")
    }

    def append(part: Arg) = parts.append(part)

    def foreach(fn: Arg => Unit): Unit = {
      parts.foreach(fn)
    }
  }

  // header that gets appended to all generated files
  val header =
    """|// SPDX-License-Identifier: Apache-2.0
       |
       |//////////////////////////////////////////////////
       |// THIS FILE WAS GENERATED, DO NOT EDIT BY HAND //
       |//////////////////////////////////////////////////
       |
       """

  /** Blocks in the templates below use this custom interpolator to produce contents
    *
    * - contiguous lines beginning with '|' are repeated only once and may only interpolate Literals
    * - contiguous lines beginning with '-' are repeated for arities 1-22 with Arity being substituted with the current arrity and Repeated being called for the range [1, current-arity]
    */
  implicit class BlockHelper(val sc: StringContext) {
    def block(args: Arg*): String = {
      val partsIterator = sc.parts.iterator
      val argsIterator = args.iterator

      // first group parts and args into lines
      val lines = ArrayBuffer[Line]()
      StringContext
        .treatEscapes(header + partsIterator.next())
        .split('\n')
        .foreach(line => lines.append(new Line(line)))
      while (partsIterator.hasNext) {
        val arg = argsIterator.next()
        val part = partsIterator.next()
        val nextLines = StringContext.treatEscapes(part).split('\n')
        lines.last.append(arg)
        lines.last.append(Literal(nextLines.head))
        nextLines.tail.foreach { line =>
          lines.append(new Line(line))
        }
      }

      // then group contiguous lines into blocks based on whether they or not they are repeated
      val blocks = ArrayBuffer(ArrayBuffer(lines.head))
      var prevLine = lines.head
      lines.tail.foreach { line =>
        if (line.isRepeated != prevLine.isRepeated) {
          blocks.append(ArrayBuffer())
        }
        blocks.last.append(line)
        prevLine = line
      }

      // finally render the blocks to a string
      val builder = new StringBuilder()
      blocks.foreach { block =>
        if (block.head.isRepeated) {
          (1 to 22).foreach { arity =>
            block.foreach { line =>
              line.foreach {
                case Literal(s) => builder ++= s
                case Arity      => builder ++= arity.toString()
                case r: Repeated =>
                  builder ++= r.start
                  (1 to arity).foreach { i =>
                    builder ++= r.fn(i)
                    if (i != arity) {
                      builder ++= r.sep
                    }
                  }
                  builder ++= r.end
              }
              builder += '\n'
            }
          }
        } else {
          block.foreach { line =>
            line.foreach {
              case Literal(s) => builder ++= s
              case arg        => throw new IllegalArgumentException(s"$arg is not allowed in a non-repeated block")
            }
            builder += '\n'
          }
        }
      }
      builder.toString()
    }
  }

  case class Template(filename: String, content: String)

  val templates = Seq(
    Template(
      "TupleProperties.scala",
      block"""|package chisel3.properties
              |
              |import chisel3.internal.{firrtl => ir}
              |import chisel3.experimental.SourceInfo
              |import firrtl.{ir => fir}
              -
              -private[chisel3] class Tuple${Arity}PropertyType[
              -  ${Repeated(i => s"_$i, PT$i <: PropertyType[_$i]")}
              -](
              -  ${Repeated(i => s"val tpe$i: PT$i")}
              -) extends TuplePropertyType[Tuple${Arity}[${Repeated(i => s"_$i")}]] {
              -  type Tuple = Tuple$Arity[${Repeated(i => s"_$i")}]
              -  type Type = Tuple$Arity[${Repeated(i => s"tpe$i.Type")}]
              -  type Underlying = Tuple$Arity[${Repeated(i => s"tpe$i.Underlying")}]
              -
              -  override def getPropertyType(): fir.PropertyType =
              -    fir.TuplePropertyType(Seq(
              -      ${Repeated(i => s"tpe$i.getPropertyType()")}
              -    ))
              -
              -  override def convert(value: Underlying, ctx: ir.Component, info: SourceInfo): fir.Expression =
              -    fir.TuplePropertyValue(Seq(
              -      ${Repeated(i => s"tpe$i.getPropertyType() -> tpe$i.convert(value._$i, ctx, info)")}
              -    ))
              -
              -  override def convertUnderlying(value: Tuple): Underlying =
              -    Tuple$Arity(
              -      ${Repeated(i => s"tpe$i.convertUnderlying(value._$i)")}
              -    )
              -}
              |
              |private[chisel3] trait LowPriorityTuplePropertyTypeInstances {
              -
              -  implicit def tuple${Arity}PropertyTypeInstance[${Repeated(i => s"_$i")}](
              -    implicit ${Repeated(i => s"tpe$i: RecursivePropertyType[_$i]")}
              -  ) = new Tuple${Arity}PropertyType[${Repeated(i => s"_$i, tpe$i.type")}](
              -    ${Repeated(i => s"tpe$i")}
              -  ) with RecursivePropertyType[Tuple$Arity[${Repeated(i => s"_$i")}]]
              |
              |}
              |
              |private[chisel3] trait TuplePropertyTypeInstances extends LowPriorityTuplePropertyTypeInstances {
              -
              -  implicit def recursiveTuple${Arity}PropertyTypeInstance[${Repeated(i => s"_$i")}](
              -    implicit ${Repeated(i => s"tpe$i: PropertyType[_$i]")}
              -  ) = new Tuple${Arity}PropertyType[${Repeated(i => s"_$i, tpe$i.type")}](
              -    ${Repeated(i => s"tpe$i")}
              -  )
              |
              |}
              |"""
    )
  )
}
