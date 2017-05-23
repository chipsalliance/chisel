// See LICENSE for license details.

package chisel3.core

import chisel3.internal.throwException

import scala.language.experimental.macros
import chisel3.internal.sourceinfo._

//scalastyle:off method.name

private[chisel3] object SeqUtils {
  /** Concatenates the data elements of the input sequence, in sequence order, together.
    * The first element of the sequence forms the least significant bits, while the last element
    * in the sequence forms the most significant bits.
    *
    * Equivalent to r(n-1) ## ... ## r(1) ## r(0).
    */
  def asUInt[T <: Bits](in: Seq[T]): UInt = macro SourceInfoTransform.inArg

  def do_asUInt[T <: Bits](in: Seq[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = {
    if (in.tail.isEmpty) {
      in.head.asUInt
    } else {
      val left = asUInt(in.slice(0, in.length/2))
      val right = asUInt(in.slice(in.length/2, in.length))
      right ## left
    }
  }

  /** Outputs the number of elements that === true.B.
    */
  def count(in: Seq[Bool]): UInt = macro SourceInfoTransform.inArg

  def do_count(in: Seq[Bool])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): UInt = in.size match {
    case 0 => 0.U
    case 1 => in.head
    case n => count(in take n/2) +& count(in drop n/2)
  }

  /** Returns the data value corresponding to the first true predicate.
    */
  def priorityMux[T <: Data](in: Seq[(Bool, T)]): T = macro SourceInfoTransform.inArg

  def do_priorityMux[T <: Data](in: Seq[(Bool, T)])
                               (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    if (in.size == 1) {
      in.head._2
    } else {
      Mux(in.head._1, in.head._2, priorityMux(in.tail))
    }
  }

  /** Returns the data value corresponding to the lone true predicate.
    * This is elaborated to firrtl using a structure that should be optimized into and and/or tree.
    *
    * @note assumes exactly one true predicate, results undefined otherwise
    *       FixedPoint values or aggregates containing FixedPoint values cause this optimized structure to be lost
    */
  def oneHotMux[T <: Data](in: Iterable[(Bool, T)]): T = macro SourceInfoTransform.inArg

  //scalastyle:off method.length cyclomatic.complexity
  def do_oneHotMux[T <: Data](in: Iterable[(Bool, T)])
                             (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    if (in.tail.isEmpty) {
      in.head._2
    }
    else {
      val output = cloneSupertype(in.toSeq map { _._2}, "oneHotMux")

      def buildAndOrMultiplexor[TT <: Data](inputs: Iterable[(Bool, TT)]): T = {
        val masked = for ((s, i) <- inputs) yield Mux(s, i.asUInt(), 0.U)
        masked.reduceLeft(_ | _).asTypeOf(output)
      }

      output match {
        case _: SInt =>
          // SInt's have to be managed carefully so sign extension works

          val sInts: Iterable[(Bool, SInt)] = in.collect { case (s: Bool, f: SInt) =>
            (s, f.asTypeOf(output).asInstanceOf[SInt])
          }

          val masked = for ((s, i) <- sInts) yield Mux(s, i, 0.S)
          masked.reduceLeft(_ | _).asTypeOf(output)

        case _: FixedPoint =>
          val (sels, possibleOuts) = in.toSeq.unzip

          val (intWidths, binaryPoints) = in.toSeq.map { case (_, o) =>
            val fo = o.asInstanceOf[FixedPoint]
            require(fo.binaryPoint.known, "Mux1H requires width/binary points to be defined")
            (fo.getWidth - fo.binaryPoint.get, fo.binaryPoint.get)
          }.unzip

          if (intWidths.distinct.length == 1 && binaryPoints.distinct.length == 1) {
            buildAndOrMultiplexor(in)
          }
          else {
            val maxIntWidth = intWidths.max
            val maxBP = binaryPoints.max
            val inWidthMatched = Seq.fill(intWidths.length)(Wire(FixedPoint((maxIntWidth + maxBP).W, maxBP.BP)))
            inWidthMatched.zipWithIndex foreach { case (e, idx) => e := possibleOuts(idx).asInstanceOf[FixedPoint] }
            buildAndOrMultiplexor(sels.zip(inWidthMatched))
          }

        case _: Aggregate =>
          val allDefineWidth = in.forall { case (_, element) => element.widthOption.isDefined }
          if(allDefineWidth) {
            buildAndOrMultiplexor(in)
          }
          else {
            throwException(s"Cannot Mux1H with aggregates with inferred widths")
          }

        case _ =>
          buildAndOrMultiplexor(in)
      }
    }
  }
}
