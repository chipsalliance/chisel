// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

abstract class OneHotEnum extends ChiselEnum with OneHotEnumIntf {
  private var next1Pos = 0

  // copied from chisel3.util
  private def isPow2(in:   BigInt): Boolean = in > 0 && ((in & (in - 1)) == 0)
  private def log2Ceil(in: BigInt): Int = (in - 1).bitLength

  override def _valIs(v: Type, lit: Type)(implicit sourceInfo: SourceInfo): Bool = {
    require(lit.isLit, "Can only compare against literal values")
    val yy = lit.litValue
    require(isPow2(yy), s"Can only compare against one-hot values, got $yy (0b${yy.toString(2)})")
    v.asUInt.apply(log2Ceil(yy))
  }

  override def _isValid(v: EnumType)(implicit sourceInfo: SourceInfo): Bool = {
    assert(v.isInstanceOf[Type])
    assert(v.getWidth == all.length, s"OneHotEnum ${this} has ${all.length} values, but ${v} has width ${v.getWidth}")
    val x = v.asUInt
    x.orR && ((x & (x - 1.U)) === 0.U)
  }

  override def _next(v: EnumType)(implicit sourceInfo: SourceInfo): v.type = {
    assert(v.isInstanceOf[Type])

    if (v.litOption.isDefined) {
      val index = v.factory.all.indexOf(v)

      if (index < v.factory.all.length - 1) {
        v.factory.all(index + 1).asInstanceOf[v.type]
      } else {
        v.factory.all.head.asInstanceOf[v.type]
      }
    } else {
      safe(v.asUInt.rotateLeft(1))._1.asInstanceOf[v.type]
    }
  }

  override def isTotal: Boolean = false

  // TODO: Is there a cleaner way?
  final implicit class OneHotType(value: Type) extends Type {
    override def isLit: Boolean = value.isLit

    override def litValue: BigInt = value.litValue

    final def is(other: Type)(implicit sourceInfo: SourceInfo): Bool = _valIs(value, other)

    /**
      * Multiplexer that selects between multiple values based on this one-hot enum.
      *
      * @param choices a sequence of tuples of (enum value, output when matched)
      * @return the output corresponding to the matched enum value
      * @note the output is undefined if none of the values match 
      */
    final def select[T <: Data](choices: Seq[(Type, T)])(implicit sourceInfo: SourceInfo): T = {
      require(choices.nonEmpty, "select must be passed a non-empty list of choices")
      // FIXME: this is a workaround to suppress a superfluous cast warning emitted by [[SeqUtils.oneHotMux]] when T is of the same EnumType. Unfortunately, it also hides potential cast warnings from the inner expressions.
      suppressEnumCastWarning {
        SeqUtils.oneHotMux(choices.map { case (oh, t) => is(oh) -> t })
      }
    }

    /**
      * Multiplexer that selects between multiple values based on this one-hot enum.
      *
      * @param firstChoice a tuple of (enum value, output when matched)
      * @param otherChoices a varargs list of tuples of (enum value, output when matched)
      * @return the output corresponding to the matched enum value
      * @note if none of the enum values match, the output is undefined
      */
    final def select[T <: Data](
      firstChoice:  (Type, T),
      otherChoices: (Type, T)*
    )(implicit sourceInfo: SourceInfo): T = select(firstChoice +: otherChoices)

  }

  def do_OHValue(name: String): Type = {
    val value = super.do_Value(name, BigInt(2).pow(next1Pos).U)
    next1Pos += 1
    value
  }
}
