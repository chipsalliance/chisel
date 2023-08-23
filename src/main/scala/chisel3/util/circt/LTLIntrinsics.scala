// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.{IntParam, IntrinsicModule, Param, StringParam}
import chisel3.experimental.hierarchy.{instantiable, public}

import circt.Intrinsic

private object Utils {
  def withoutNone(params: Map[String, Option[Param]]): Map[String, Param] =
    params.collect { case (name, Some(param)) => (name, param) }
}

@instantiable
class BaseIntrinsic(
  intrinsicName: String,
  maybeParams:   Map[String, Option[Param]] = Map.empty[String, Option[Param]])
    extends IntrinsicModule(f"circt_$intrinsicName", Utils.withoutNone(maybeParams)) {
  private val paramValues = params.values.map(_ match {
    case IntParam(x)    => x.toString
    case StringParam(x) => x.toString
    case x              => x.toString
  })
  override def desiredName = paramValues.fold(this.getClass.getSimpleName) { (a, b) => a + "_" + b }
}

// Base class for all unary intrinsics with `in` and `out` ports. */
@instantiable
class UnaryLTLIntrinsic(
  intrinsicName: String,
  params:        Map[String, Option[Param]] = Map.empty[String, Option[Param]])
    extends BaseIntrinsic(f"ltl_$intrinsicName", params) {
  @public val in = IO(Input(Bool()))
  @public val out = IO(Output(Bool()))
}

// Base class for all binary intrinsics with `lhs`, `rhs`, and `out` ports. */
@instantiable
class BinaryLTLIntrinsic(
  intrinsicName: String,
  params:        Map[String, Option[Param]] = Map.empty[String, Option[Param]])
    extends BaseIntrinsic(f"ltl_$intrinsicName", params) {
  @public val lhs = IO(Input(Bool()))
  @public val rhs = IO(Input(Bool()))
  @public val out = IO(Output(Bool()))
}

// A wrapper intrinsic for the CIRCT `ltl.and` operation. */
class LTLAndIntrinsic extends BinaryLTLIntrinsic("and")

// A wrapper intrinsic for the CIRCT `ltl.or` operation. */
class LTLOrIntrinsic extends BinaryLTLIntrinsic("or")

// A wrapper intrinsic for the CIRCT `ltl.delay` operation. */
class LTLDelayIntrinsic(delay: Int, length: Option[Int])
    extends UnaryLTLIntrinsic("delay", Map("delay" -> Some(IntParam(delay)), "length" -> length.map(IntParam(_))))

// A wrapper intrinsic for the CIRCT `ltl.concat` operation. */
class LTLConcatIntrinsic extends BinaryLTLIntrinsic("concat")

// A wrapper intrinsic for the CIRCT `ltl.not` operation. */
class LTLNotIntrinsic extends UnaryLTLIntrinsic("not")

// A wrapper intrinsic for the CIRCT `ltl.implication` operation. */
class LTLImplicationIntrinsic extends BinaryLTLIntrinsic("implication")

// A wrapper intrinsic for the CIRCT `ltl.eventually` operation. */
class LTLEventuallyIntrinsic extends UnaryLTLIntrinsic("eventually")

// A wrapper intrinsic for the CIRCT `ltl.clock` operation. */
@instantiable
class LTLClockIntrinsic extends BaseIntrinsic("ltl_clock") {
  @public val in = IO(Input(Bool()))
  @public val clock = IO(Input(Clock()))
  @public val out = IO(Output(Bool()))
}

// A wrapper intrinsic for the CIRCT `ltl.disable` operation. */
@instantiable
class LTLDisableIntrinsic extends BaseIntrinsic("ltl_disable") {
  @public val in = IO(Input(Bool()))
  @public val condition = IO(Input(Bool()))
  @public val out = IO(Output(Bool()))
}

// Base class for assert, assume, and cover intrinsics.
@instantiable
class VerifAssertLikeIntrinsic(intrinsicName: String, label: Option[String])
    extends BaseIntrinsic(f"verif_$intrinsicName", Map("label" -> label.map(StringParam(_)))) {
  @public val property = IO(Input(Bool()))
}

// A wrapper intrinsic for the CIRCT `verif.assert` operation.
class VerifAssertIntrinsic(label: Option[String]) extends VerifAssertLikeIntrinsic("assert", label)

// A wrapper intrinsic for the CIRCT `verif.assume` operation.
class VerifAssumeIntrinsic(label: Option[String]) extends VerifAssertLikeIntrinsic("assume", label)

// A wrapper intrinsic for the CIRCT `verif.cover` operation.
class VerifCoverIntrinsic(label: Option[String]) extends VerifAssertLikeIntrinsic("cover", label)
