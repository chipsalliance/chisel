// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.ltl.Property
import chisel3.experimental.{IntParam, IntrinsicModule, Param, StringParam}
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.SourceInfo

/** Base instrinsic for circt related intrinsics
  * @param intrinsicName name of the intrinsic
  * @param ret return type of the expression
  * @param params parameter name/value pairs, if any.  Parameter names must be unique.
  * @param sourceInfo where the data is from
  * @return intrinsic expression that returns the specified return type
  */
private[chisel3] object BaseIntrinsic {
  def apply[T <: Data](
    intrinsicName: String,
    ret:           => T,
    maybeParams:   Seq[(String, Param)] = Seq()
  )(data:          Data*
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    IntrinsicExpr(f"circt_$intrinsicName", ret, maybeParams: _*)(data: _*)
  }
}

/** Base intrinsic for all unary intrinsics with `in` and `out` ports. */
private[chisel3] object UnaryLTLIntrinsic {
  def apply[T <: Data](
    intrinsicName: String,
    params:        Seq[(String, Param)] = Seq()
  )(_in:           T
  )(
    implicit sourceInfo: SourceInfo
  ): Bool =
    BaseIntrinsic(f"ltl_$intrinsicName", Bool(), params)(_in)
}

/** Base instrinsic for all binary intrinsics with `lhs`, `rhs`, and `out` ports. */
private[chisel3] object BinaryLTLIntrinsic {
  def apply[T <: Data, U <: Data](
    intrinsicName: String,
    params:        Seq[(String, Param)] = Seq()
  )(lhs:           T,
    rhs:           U
  )(
    implicit sourceInfo: SourceInfo
  ): Bool =
    BaseIntrinsic(f"ltl_$intrinsicName", Bool(), params)(lhs, rhs)
}

/** A wrapper intrinsic for the CIRCT `ltl.and` operation. */
private[chisel3] object LTLAndIntrinsic {
  def apply(lhs: Bool, rhs: Bool)(implicit sourceInfo: SourceInfo) =
    BinaryLTLIntrinsic("and")(lhs, rhs)
}

/** A wrapper intrinsic for the CIRCT `ltl.or` operation. */
private[chisel3] object LTLOrIntrinsic {
  def apply(lhs: Bool, rhs: Bool)(implicit sourceInfo: SourceInfo) =
    BinaryLTLIntrinsic("or")(lhs, rhs)
}

/** A wrapper intrinsic for the CIRCT `ltl.intersect` operation. */
private[chisel3] object LTLIntersectIntrinsic {
  def apply(lhs: Bool, rhs: Bool)(implicit sourceInfo: SourceInfo) =
    BinaryLTLIntrinsic("intersect")(lhs, rhs)
}

/** A wrapper intrinsic for the CIRCT `ltl.until` operation. */
private[chisel3] object LTLUntilIntrinsic {
  def apply(lhs: Bool, rhs: Bool)(implicit sourceInfo: SourceInfo) =
    BinaryLTLIntrinsic("until")(lhs, rhs)
}

/** A wrapper intrinsic for the CIRCT `ltl.delay` operation. */
private[chisel3] object LTLDelayIntrinsic {

  def apply(delay: Int, length: Option[Int])(_in: Bool)(implicit sourceInfo: SourceInfo) = {
    val params = Seq("delay" -> IntParam(delay)) ++ length.map("length" -> IntParam(_))
    UnaryLTLIntrinsic("delay", params)(_in)
  }
}

/** A wrapper intrinsic for the CIRCT `ltl.repeat` operation. */
private[chisel3] object LTLRepeatIntrinsic {

  def apply(base: Int, more: Option[Int])(_in: Bool)(implicit sourceInfo: SourceInfo) = {
    val params = Seq("base" -> IntParam(base)) ++ more.map("more" -> IntParam(_))
    UnaryLTLIntrinsic("repeat", params)(_in)
  }
}

/** A wrapper intrinsic for the CIRCT `ltl.goto_repeat` operation. */
private[chisel3] object LTLGoToRepeatIntrinsic {
  def apply(base: Int, more: Int)(_in: Bool)(implicit sourceInfo: SourceInfo) =
    UnaryLTLIntrinsic("goto_repeat", Seq("base" -> IntParam(base), "more" -> IntParam(more)))(_in)
}

/** A wrapper intrinsic for the CIRCT `ltl.non_consecutive_repeat` operation. */
private[chisel3] object LTLNonConsecutiveRepeatIntrinsic {
  def apply(base: Int, more: Int)(_in: Bool)(implicit sourceInfo: SourceInfo) =
    UnaryLTLIntrinsic("non_consecutive_repeat", Seq("base" -> IntParam(base), "more" -> IntParam(more)))(_in)
}

/** A wrapper intrinsic for the CIRCT `ltl.concat` operation. */
private[chisel3] object LTLConcatIntrinsic {
  def apply(lhs: Bool, rhs: Bool)(implicit sourceInfo: SourceInfo) =
    BinaryLTLIntrinsic("concat")(lhs, rhs)
}

/** A wrapper intrinsic for the CIRCT `ltl.not` operation. */
private[chisel3] object LTLNotIntrinsic {
  def apply(prop: Bool)(implicit sourceInfo: SourceInfo) =
    UnaryLTLIntrinsic("not")(prop)
}

/** A wrapper intrinsic for the CIRCT `ltl.implication` operation. */
private[chisel3] object LTLImplicationIntrinsic {
  def apply(antecedent: Bool, consequent: Bool)(implicit sourceInfo: SourceInfo) =
    BinaryLTLIntrinsic("implication")(antecedent, consequent)
}

/** A wrapper intrinsic for the CIRCT `ltl.eventually` operation. */
private[chisel3] object LTLEventuallyIntrinsic {
  def apply(prop: Bool)(implicit sourceInfo: SourceInfo) =
    UnaryLTLIntrinsic("eventually")(prop)
}

/** A wrapper intrinsic for the CIRCT `ltl.clock` operation. */
private[chisel3] object LTLClockIntrinsic {
  def apply(in: Bool, clock: Clock)(implicit sourceInfo: SourceInfo) =
    BinaryLTLIntrinsic("clock")(in, clock)
}

/** A wrapper intrinsic for the CIRCT `ltl.disable` operation. */
private[chisel3] object LTLDisableIntrinsic {
  def apply(in: Bool, cond: Bool)(implicit sourceInfo: SourceInfo) =
    BinaryLTLIntrinsic("disable")(in, cond)
}

/** Base class for assert, assume, and cover intrinsics. */
private[chisel3] object VerifAssertLikeIntrinsic {
  def apply(
    intrinsicName: String,
    label:         Option[String]
  )(prop:          Bool,
    enable:        Option[Bool]
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    val name = f"circt_verif_$intrinsicName"
    Intrinsic(name, (label.map("label" -> StringParam(_)).toSeq): _*)((Seq(prop) ++ enable.toSeq): _*)
  }
}

/** A wrapper intrinsic for the CIRCT `verif.assert` operation. */
private[chisel3] object VerifAssertIntrinsic {
  def apply(label: Option[String] = None)(prop: Bool, enable: Option[Bool])(implicit sourceInfo: SourceInfo) =
    VerifAssertLikeIntrinsic("assert", label)(prop, enable)
}

/** A wrapper intrinsic for the CIRCT `verif.assume` operation. */
private[chisel3] object VerifAssumeIntrinsic {
  def apply(label: Option[String] = None)(prop: Bool, enable: Option[Bool])(implicit sourceInfo: SourceInfo) =
    VerifAssertLikeIntrinsic("assume", label)(prop, enable)
}

/** A wrapper intrinsic for the CIRCT `verif.cover` operation. */
private[chisel3] object VerifCoverIntrinsic {
  def apply(label: Option[String] = None)(prop: Bool, enable: Option[Bool])(implicit sourceInfo: SourceInfo) =
    VerifAssertLikeIntrinsic("cover", label)(prop, enable)
}
