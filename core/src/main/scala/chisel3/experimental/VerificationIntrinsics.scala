package chisel3.util.circt

import chisel3._
import chisel3.internal._
import chisel3.experimental.{BaseModule, IntrinsicModule, SourceInfo}

/** Create a module for `ifelsefatal` style assertion.
  */
private[chisel3] class IfElseFatalIntrinsic[T <: Data](
  implicit sourceInfo: SourceInfo,
  id:                  BaseModule,
  format:              Printable,
  label:               String,
  data:                Seq[T])
    extends IntrinsicModule(
      "circt_chisel_ifelsefatal",
      Map("format" -> chisel3.experimental.PrintableParam(format, id), "label" -> label)
    ) {
  // Because this code is in core, the Chisel compiler plugin does not run on it so we must .suggestName
  val clock = IO(Input(Clock())).suggestName("clock")
  val predicate = IO(Input(Bool())).suggestName("predicate")
  val enable = IO(Input(Bool())).suggestName("enable")
  val args = data.zipWithIndex.map({
    case (d, i) =>
      IO(Input(chiselTypeOf(d)).suggestName(s"args_${i}"))
  })
}
