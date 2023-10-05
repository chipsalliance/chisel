// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{prefix, SourceInfo}
import chisel3.internal.throwException

import scala.language.experimental.macros
import chisel3.internal.sourceinfo._
import chisel3.internal.plugin.autoNameRecursively

private[chisel3] object SeqUtils {

  /** Concatenates the data elements of the input sequence, in sequence order, together.
    * The first element of the sequence forms the least significant bits, while the last element
    * in the sequence forms the most significant bits.
    *
    * Equivalent to r(n-1) ## ... ## r(1) ## r(0).
    * @note This returns a `0.U` if applied to a zero-element `Vec`.
    */
  def asUInt[T <: Bits](in: Seq[T]): UInt = macro SourceInfoTransform.inArg

  /** @group SourceInfoTransformMacros */
  def do_asUInt[T <: Bits](in: Seq[T])(implicit sourceInfo: SourceInfo): UInt = {
    if (in.isEmpty) {
      0.U
    } else if (in.tail.isEmpty) {
      in.head.asUInt
    } else {
      val lo = autoNameRecursively("lo")(prefix("lo") {
        asUInt(in.slice(0, in.length / 2))
      })
      val hi = autoNameRecursively("hi")(prefix("hi") {
        asUInt(in.slice(in.length / 2, in.length))
      })
      hi ## lo
    }
  }

  /** Outputs the number of elements that === true.B.
    */
  def count(in: Seq[Bool]): UInt = macro SourceInfoTransform.inArg

  /** @group SourceInfoTransformMacros */
  def do_count(in: Seq[Bool])(implicit sourceInfo: SourceInfo): UInt = in.size match {
    case 0 => 0.U
    case 1 => in.head
    case n =>
      val sum = count(in.take(n / 2)) +& count(in.drop(n / 2))
      sum(BigInt(n).bitLength - 1, 0)
  }

  /** Returns the data value corresponding to the first true predicate.
    */
  def priorityMux[T <: Data](in: Seq[(Bool, T)]): T = macro SourceInfoTransform.inArg

  /** @group SourceInfoTransformMacros */
  def do_priorityMux[T <: Data](
    in: Seq[(Bool, T)]
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    if (in.size == 1) {
      in.head._2
    } else {
      val r = in.view.reverse
      r.tail.foldLeft(r.head._2) {
        case (alt, (sel, elt)) => Mux(sel, elt, alt)
      }
    }
  }

  /** Returns the data value corresponding to the lone true predicate.
    * This is elaborated to firrtl using a structure that should be optimized into and and/or tree.
    *
    * @note assumes exactly one true predicate, results undefined otherwise
    */
  def oneHotMux[T <: Data](in: Iterable[(Bool, T)]): T = macro SourceInfoTransform.inArg

  /** @group SourceInfoTransformMacros */
  def do_oneHotMux[T <: Data](
    in: Iterable[(Bool, T)]
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    if (in.tail.isEmpty) {
      in.head._2
    } else {
      val output = cloneSupertype(in.toSeq.map { _._2 }, "oneHotMux")

      def buildAndOrMultiplexor[TT <: Data](inputs: Iterable[(Bool, TT)]): T = {
        val masked = for ((s, i) <- inputs) yield Mux(s, i.asUInt, 0.U)
        masked.reduceLeft(_ | _).asTypeOf(output)
      }

      output match {
        case _: SInt =>
          // SInt's have to be managed carefully so sign extension works

          val sInts: Iterable[(Bool, SInt)] = in.collect {
            case (s: Bool, f: SInt) =>
              (s, f.asTypeOf(output).asInstanceOf[SInt])
          }

          val masked = for ((s, i) <- sInts) yield Mux(s, i, 0.S)
          masked.reduceLeft(_ | _).asTypeOf(output)

        case agg: Aggregate =>
          val allDefineWidth = in.forall { case (_, element) => element.widthOption.isDefined }
          if (allDefineWidth) {
            val out = Wire(agg)
            val (sel, inData) = in.unzip
            val inElts = inData.map(_.asInstanceOf[Aggregate].getElements)
            // We want to iterate on the columns of inElts, so we transpose
            out.getElements.zip(inElts.transpose).foreach {
              case (outElt, elts) =>
                outElt := oneHotMux(sel.zip(elts))
            }
            out.asInstanceOf[T]
          } else {
            throwException(s"Cannot Mux1H with aggregates with inferred widths")
          }

        case _ =>
          buildAndOrMultiplexor(in)
      }
    }
  }
}
