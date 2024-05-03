// SPDX-License-Identifier: Apache-2.0

package chisel3.util.circt

import chisel3._
import chisel3.experimental.{IntParam, IntrinsicModule, Param, StringParam}
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.SourceInfo

/*private object Utils {
  private[chisel3] def withoutNone(params: Map[String, Option[Param]]): Map[String, Param] =
    params.collect { case (name, Some(param)) => (name, param) }
}*/

/*@instantiable
private[chisel3] class BaseIntrinsic(
  intrinsicName: String,
  maybeParams:   Map[String, Option[Param]] = Map.empty[String, Option[Param]])
    extends IntrinsicModule(f"circt_$intrinsicName", Utils.withoutNone(maybeParams)) {
  private val paramValues = params.values.map(_ match {
    case IntParam(x)    => x.toString
    case StringParam(x) => x.toString
    case x              => x.toString
  })
  override def desiredName = paramValues.fold(this.getClass.getSimpleName) { (a, b) => a + "_" + b }
}*/

/** Base instrinsic for circt related intrinsics
  * @param intrinsicName name of the intrinsic
  * @param ret return type of the expression
  * @param params parameter name/value pairs, if any.  Parameter names must be unique.
  * @param sourceInfo where the data is from
  * @return intrinsic expression that returns the specified return type
  */
object BaseIntrinsic {
  def apply[T <: Data](
    intrinsicName: String,
    ret: => T,
    maybeParams:   Seq[(String, Param)] = Seq()
  )(implicit sourceInfo: SourceInfo) = {
    IntrinsicExpr(f"circt_$intrinsicName", ret, maybeParams:_*)()
  }
}

/*/** Base class for all unary intrinsics with `in` and `out` ports. */
@instantiable
private[chisel3] class UnaryLTLIntrinsic(
  intrinsicName: String,
  params:        Map[String, Option[Param]] = Map.empty[String, Option[Param]])
    extends BaseIntrinsic(f"ltl_$intrinsicName", params) {
  @public val in = IO(Input(Bool()))
  @public val out = IO(Output(Bool()))
}*/

/** Base intrinsic for all unary intrinsics with `in` and `out` ports. */
object UnaryLTLIntrinsic {
  def apply[T <: Data](
    intrinsicName: String,
    params:   Seq[(String, Param)] = Seq()
  )(implicit sourceInfo: SourceInfo) = {

    class UnaryLTLIntrinsicBundle extends Bundle {
      val in = IO(Input(Bool()))
      val out = IO(Output(Bool()))
    }

    BaseIntrinsic(f"ltl_$intrinsicName", new UnaryLTLIntrinsicBundle, params)
  }
}

/** Base class for all binary intrinsics with `lhs`, `rhs`, and `out` ports. */
/*@instantiable
private[chisel3] class BinaryLTLIntrinsic(
  intrinsicName: String,
  params:        Map[String, Option[Param]] = Map.empty[String, Option[Param]])
    extends BaseIntrinsic(f"ltl_$intrinsicName", params) {
  @public val lhs = Input(Bool())
  @public val rhs = Input(Bool())
  @public val out = Output(Bool())
}*/

/** Base instrinsic for all binary intrinsics with `lhs`, `rhs`, and `out` ports. */
object BinaryLTLIntrinsic {
  def apply[T <: Data](
    intrinsicName: String,
    params:   Seq[(String, Param)] = Seq()
  )(implicit sourceInfo: SourceInfo) = {

    class BinaryLTLIntrinsicBundle extends Bundle {
      val lhs = Input(Bool())
      val rhs = Input(Bool())
      val out = Output(Bool())
    }

    BaseIntrinsic(f"ltl_$intrinsicName", new BinaryLTLIntrinsicBundle, params)
  }
}

/** A wrapper intrinsic for the CIRCT `ltl.and` operation. */
object LTLAndIntrinsic {
  def apply(implicit sourceInfo: SourceInfo) = BinaryLTLIntrinsic("and")
}

/** A wrapper intrinsic for the CIRCT `ltl.or` operation. */
object LTLOrIntrinsic {
  def apply(implicit sourceInfo: SourceInfo) = BinaryLTLIntrinsic("or")
}

/*private[chisel3] class LTLDelayIntrinsic(delay: Int, length: Option[Int])
extends UnaryLTLIntrinsic("delay", Map("delay" -> Some(IntParam(delay)), "length" -> length.map(IntParam(_))))*/

/** A wrapper intrinsic for the CIRCT `ltl.delay` operation. */
object LTLDelayIntrinsic {
  
  def apply(delay: Int, length: Option[Int])(implicit sourceInfo: SourceInfo) = {
    val params = length match {
      case None => Seq("delay" -> IntParam(delay))
      case Some(l) => Seq("delay" -> IntParam(delay), "length" -> IntParam(l))
    }
    UnaryLTLIntrinsic("delay", params)
  }
}

/** A wrapper intrinsic for the CIRCT `ltl.concat` operation. */
/*private[chisel3] class LTLConcatIntrinsic extends BinaryLTLIntrinsic("concat")*/
object LTLConcatIntrinsic {
  def apply(implicit sourceInfo: SourceInfo) = BinaryLTLIntrinsic("concat")
}

/** A wrapper intrinsic for the CIRCT `ltl.not` operation. */
object LTLNotIntrinsic {
  def apply(implicit sourceInfo: SourceInfo) = UnaryLTLIntrinsic("not")
}
//private[chisel3] class LTLNotIntrinsic extends UnaryLTLIntrinsic("not")

/** A wrapper intrinsic for the CIRCT `ltl.implication` operation. */
object LTLImplicationIntrinsic {
  def apply(implicit sourceInfo: SourceInfo) = BinaryLTLIntrinsic("implication")
}
//private[chisel3] class LTLImplicationIntrinsic extends BinaryLTLIntrinsic("implication")

/** A wrapper intrinsic for the CIRCT `ltl.eventually` operation. */
object LTLEventuallyIntrinsic {
  def apply(implicit sourceInfo: SourceInfo) = UnaryLTLIntrinsic("eventually")
}
//private[chisel3] class LTLEventuallyIntrinsic extends UnaryLTLIntrinsic("eventually")

/** A wrapper intrinsic for the CIRCT `ltl.clock` operation. */
object LTLClockIntrinsic {
  def apply(implicit sourceInfo: SourceInfo) = BaseIntrinsic(
    "ltl_clock", 
    new Bundle {
      val in = Input(Bool())
      val clock = Input(Clock())
      val out = Output(Bool())
    }
  )
}
/*@instantiable
private[chisel3] class LTLClockIntrinsic extends BaseIntrinsic("ltl_clock") {
  @public val in = IO(Input(Bool()))
  @public val clock = IO(Input(Clock()))
  @public val out = IO(Output(Bool()))
}*/

/** A wrapper intrinsic for the CIRCT `ltl.disable` operation. */
object LTLDisableIntrinsic {
  def apply(implicit sourceInfo: SourceInfo) = BaseIntrinsic(
    "ltl_disable", 
    new Bundle {
      val in = Input(Bool())
      val condition = Input(Bool())
      val out = Output(Bool())
    }
  )
}
/*@instantiable
private[chisel3] class LTLDisableIntrinsic extends BaseIntrinsic("ltl_disable") {
  @public val in = IO(Input(Bool()))
  @public val condition = IO(Input(Bool()))
  @public val out = IO(Output(Bool()))
}*/

/** Base class for assert, assume, and cover intrinsics. */
object VerifAssertLikeIntrinsic {
  def apply(intrinsicName: String, label: Option[String])(implicit sourceInfo: SourceInfo) = {
    val name = f"verif_$intrinsicName"
    val io = new Bundle { val property = Input(Bool()) }

    label match {
      case None => BaseIntrinsic(name, io)
      case Some(l) => BaseIntrinsic(name, io, Seq("label" -> StringParam(l)))
    }
  }
}

/*@instantiable
private[chisel3] class VerifAssertLikeIntrinsic(intrinsicName: String, label: Option[String])
    extends BaseIntrinsic(f"verif_$intrinsicName", Map("label" -> label.map(StringParam(_)))) {
  @public val property = IO(Input(Bool()))
}*/

/** A wrapper intrinsic for the CIRCT `verif.assert` operation. */
object VerifAssertIntrinsic {
  def apply(label: Option[String] = None)(implicit sourceInfo: SourceInfo) = VerifAssertLikeIntrinsic("assert", label)
}
/*private[chisel3] class VerifAssertIntrinsic(label: Option[String]) extends VerifAssertLikeIntrinsic("assert", label)*/

/** A wrapper intrinsic for the CIRCT `verif.assume` operation. */
object VerifAssumeIntrinsic {
  def apply(label: Option[String] = None)(implicit sourceInfo: SourceInfo) = VerifAssertLikeIntrinsic("assume", label)
}
//private[chisel3] class VerifAssumeIntrinsic(label: Option[String]) extends VerifAssertLikeIntrinsic("assume", label)

/** A wrapper intrinsic for the CIRCT `verif.cover` operation. */
object VerifCoverIntrinsic {
  def apply(label: Option[String] = None)(implicit sourceInfo: SourceInfo) = VerifAssertLikeIntrinsic("cover", label)
}
//private[chisel3] class VerifCoverIntrinsic(label: Option[String]) extends VerifAssertLikeIntrinsic("cover", label)

// There appears to be a bug in ScalaDoc where you cannot use macro-generated methods in the same
// compilation unit as the macro-generated type. This means that any use of Definition[_] or
// Instance[_] of the above classes within this compilation unit breaks ScalaDoc generation. This
// issue appears to be similar to https://stackoverflow.com/questions/42684101 but applying the
// specific mitigation did not seem to work.  As a workaround, we simply write the extension methods
// that are generated by the @instantiable macro so that we can use them here.
/*private[chisel3] object LTLIntrinsicInstanceMethodsInternalWorkaround {
  import chisel3.experimental.hierarchy.Instance
  implicit val mg: chisel3.internal.MacroGenerated = new chisel3.internal.MacroGenerated {}
  implicit class UnaryLTLIntrinsicInstanceMethods(underlying: Instance[UnaryLTLIntrinsic]) {
    def in = underlying._lookup(_.in)
    def out = underlying._lookup(_.out)
  }
  implicit class BinaryLTLIntrinsicInstanceMethods(underlying: Instance[BinaryLTLIntrinsic]) {
    def lhs = underlying._lookup(_.lhs)
    def rhs = underlying._lookup(_.rhs)
    def out = underlying._lookup(_.out)
  }
  implicit class LTLClockIntrinsicInstanceMethods(underlying: Instance[LTLClockIntrinsic]) {
    def in = underlying._lookup(_.in)
    def clock = underlying._lookup(_.clock)
    def out = underlying._lookup(_.out)
  }
  implicit class LTLDisableIntrinsicInstanceMethods(underlying: Instance[LTLDisableIntrinsic]) {
    def in = underlying._lookup(_.in)
    def condition = underlying._lookup(_.condition)
    def out = underlying._lookup(_.out)
  }
  implicit class VerifAssertLikeIntrinsicInstanceMethods(underlying: Instance[VerifAssertLikeIntrinsic]) {
    def property = underlying._lookup(_.property)
  }
}*/
