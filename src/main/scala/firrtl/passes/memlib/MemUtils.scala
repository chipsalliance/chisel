// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.PrimOps._

/** Given a mask, return a bitmask corresponding to the desired datatype.
 *  Requirements:
 *    - The mask type and datatype must be equivalent, except any ground type in
 *         datatype must be matched by a 1-bit wide UIntType.
 *    - The mask must be a reference, subfield, or subindex
 *  The bitmask is a series of concatenations of the single mask bit over the
 *    length of the corresponding ground type, e.g.:
 *{{{
 * wire mask: {x: UInt<1>, y: UInt<1>}
 * wire data: {x: UInt<2>, y: SInt<2>}
 * // this would return:
 * cat(cat(mask.x, mask.x), cat(mask.y, mask.y))
 * }}}
 */
object toBitMask {
  def apply(mask: Expression, dataType: Type): Expression = mask match {
    case ex @ (_: WRef | _: WSubField | _: WSubIndex) => hiermask(ex, dataType)
    case t => error("Invalid operand expression for toBits!")
  }
  private def hiermask(mask: Expression, dataType: Type): Expression =
    (mask.tpe, dataType) match {
      case (mt: VectorType, dt: VectorType) =>
        seqCat((0 until mt.size).reverse map { i =>
          hiermask(WSubIndex(mask, i, mt.tpe, UNKNOWNGENDER), dt.tpe)
        })
      case (mt: BundleType, dt: BundleType) =>
        seqCat((mt.fields zip dt.fields) map { case (mf, df) =>
          hiermask(WSubField(mask, mf.name, mf.tpe, UNKNOWNGENDER), df.tpe)
        })
      case (UIntType(width), dt: GroundType) if width == IntWidth(BigInt(1)) =>
        seqCat(List.fill(bitWidth(dt).intValue)(mask))
      case (mt, dt) => error("Invalid type for mask component!")
    }
}

object createMask {
  def apply(dt: Type): Type = dt match {
    case t: VectorType => VectorType(apply(t.tpe), t.size)
    case t: BundleType => BundleType(t.fields map (f => f copy (tpe=apply(f.tpe))))
    case GroundType(w) if w == IntWidth(0) => UIntType(IntWidth(0))
    case t: GroundType => BoolType
  }
}

object MemPortUtils {
  type MemPortMap = collection.mutable.HashMap[String, Expression]
  type Memories = collection.mutable.ArrayBuffer[DefMemory]
  type Modules = collection.mutable.ArrayBuffer[DefModule]

  def defaultPortSeq(mem: DefMemory): Seq[Field] = Seq(
    Field("addr", Default, UIntType(IntWidth(ceilLog2(mem.depth) max 1))),
    Field("en", Default, BoolType),
    Field("clk", Default, ClockType)
  )

  // Todo: merge it with memToBundle
  def memType(mem: DefMemory): Type = {
    val rType = BundleType(defaultPortSeq(mem) :+
      Field("data", Flip, mem.dataType))
    val wType = BundleType(defaultPortSeq(mem) ++ Seq(
      Field("data", Default, mem.dataType),
      Field("mask", Default, createMask(mem.dataType))))
    val rwType = BundleType(defaultPortSeq(mem) ++ Seq(
      Field("rdata", Flip, mem.dataType),
      Field("wmode", Default, BoolType),
      Field("wdata", Default, mem.dataType),
      Field("wmask", Default, createMask(mem.dataType))))
    BundleType(
      (mem.readers map (Field(_, Flip, rType))) ++
      (mem.writers map (Field(_, Flip, wType))) ++
      (mem.readwriters map (Field(_, Flip, rwType))))
  }

  def memPortField(s: DefMemory, p: String, f: String): Expression = {
    val mem = WRef(s.name, memType(s), MemKind, UNKNOWNGENDER)
    val t1 = field_type(mem.tpe, p)
    val t2 = field_type(t1, f)
    WSubField(WSubField(mem, p, t1, UNKNOWNGENDER), f, t2, UNKNOWNGENDER)
  }
}
