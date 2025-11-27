package chisel3.util.circt

import chisel3._
import chisel3.internal._
import chisel3.experimental.{fromStringToStringParam, BaseModule, SourceInfo}

/** Create an `ifelsefatal` style assertion.
  */
private[chisel3] object IfElseFatalIntrinsic {
  def apply(
    id:        BaseModule,
    format:    Printable,
    label:     String,
    clock:     Clock,
    predicate: Bool,
    enable:    Bool,
    data:      Data*
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    Intrinsic(
      "circt_chisel_ifelsefatal",
      "format" -> chisel3.PrintableParam(format, id),
      "label" -> label
    )((Seq(clock, predicate, enable) ++ data): _*)
  }
}
