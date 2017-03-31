// See LICENSE for license details.

package chisel3.core

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

  def do_asUInt[T <: Bits](in: Seq[T])(implicit sourceInfo: SourceInfo): UInt = {
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

  def do_count(in: Seq[Bool])(implicit sourceInfo: SourceInfo): UInt = in.size match {
    case 0 => 0.U
    case 1 => in.head
    case n => count(in take n/2) +& count(in drop n/2)
  }

  /** Returns the data value corresponding to the first true predicate.
    */
  def priorityMux[T <: Data](in: Seq[(Bool, T)]): T = macro CompileOptionsTransform.inArg

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
  def oneHotMux[T <: Data](in: Iterable[(Bool, T)]): T = macro CompileOptionsTransform.inArg

  //scalastyle:off method.length cyclomatic.complexity
  def do_oneHotMux[T <: Data](in: Iterable[(Bool, T)])
                             (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
    if (in.tail.isEmpty) {
      in.head._2
    }
    else {
      val compares = in.map { case (_, element) => in.head._2.typeEquivalent(element) }
      val allSame = in.forall { case (_, element) => in.head._2.typeEquivalent(element) }
      if(allSame) {
        val output = cloneSupertype(in.toSeq map {_._2}, "oneHotMux")
        val masked = for ((s, i) <- in) yield Mux(s, i.asUInt(), 0.U)
        output.fromBits(masked.reduceLeft(_|_))
      }
      in.head._2 match {
        case _: SInt =>
          // SInt's have to be managed carefully so sign extension works
          val output = cloneSupertype(in.toSeq map {_._2}, "oneHotMux")

          val sInts: Iterable[(Bool, SInt)] = in.collect { case (s: Bool, f: SInt) =>
            (s, f.asTypeOf(output).asInstanceOf[SInt])
          }

          val masked = for ((s, i) <- sInts) yield Mux(s, i, 0.S)

          output.fromBits(masked.reduceLeft(_|_))

        case _: FixedPoint =>
          // Fixed point cannot use the standard methodology here
          val output = cloneSupertype(in.toSeq map {_._2}, "oneHotMux")

          //          val masked = in.collect { case (s: Bool, f: FixedPoint) => (s, f) }.map { case (s, f) =>
          //            Mux(
          //              s, f, 0.U.asTypeOf(output).asInstanceOf[FixedPoint])
          //          }

          val fixedPoints = in.collect { case (s: Bool, f: FixedPoint) => (s, f) }
          def makeMux(inputs: List[(Bool, FixedPoint)]): FixedPoint = {
            inputs match {
              case Nil => 0.U.asTypeOf(output).asInstanceOf[FixedPoint]
              case (select, value) :: tail  =>
                Mux(select, value, makeMux(tail))
            }
          }

          makeMux(fixedPoints.toList).asTypeOf(output)
        //          val masked = for ((s, i) <- in)
        //            yield Mux(
        //              s, i.asTypeOf(output).asInstanceOf[FixedPoint], 0.U.asTypeOf(output).asInstanceOf[FixedPoint])
        case aggregate: Aggregate =>
          val output = cloneSupertype(in.toSeq map {_._2}, "oneHotMux")
          val masked = for ((s, i) <- in) yield Mux(s, i.asUInt(), 0.U)
          output.fromBits(masked.reduceLeft(_|_))
        case _ =>
          val output = cloneSupertype(in.toSeq map {_._2}, "oneHotMux")
          val masked = for ((s, i) <- in) yield Mux(s, i.asUInt(), 0.U)
          output.fromBits(masked.reduceLeft(_|_))
      }
    }
  }
}
