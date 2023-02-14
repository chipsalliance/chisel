// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._

object createMask {
  def apply(dt: Type): Type = dt match {
    case t: VectorType => VectorType(apply(t.tpe), t.size)
    case t: BundleType => BundleType(t.fields.map(f => f.copy(tpe = apply(f.tpe))))
    case GroundType(w) if w == IntWidth(0) => UIntType(IntWidth(0))
    case t: GroundType => BoolType
  }
}

object MemPortUtils {
  type MemPortMap = collection.mutable.HashMap[String, Expression]
  type Memories = collection.mutable.ArrayBuffer[DefMemory]
  type Modules = collection.mutable.ArrayBuffer[DefModule]

  def defaultPortSeq(mem: DefMemory): Seq[Field] = Seq(
    Field("addr", Default, UIntType(IntWidth(getUIntWidth(mem.depth - 1).max(1)))),
    Field("en", Default, BoolType),
    Field("clk", Default, ClockType)
  )

  // Todo: merge it with memToBundle
  def memType(mem: DefMemory): BundleType = {
    val rType = BundleType(
      defaultPortSeq(mem) :+
        Field("data", Flip, mem.dataType)
    )
    val wType = BundleType(
      defaultPortSeq(mem) ++ Seq(Field("data", Default, mem.dataType), Field("mask", Default, createMask(mem.dataType)))
    )
    val rwType = BundleType(
      defaultPortSeq(mem) ++ Seq(
        Field("rdata", Flip, mem.dataType),
        Field("wmode", Default, BoolType),
        Field("wdata", Default, mem.dataType),
        Field("wmask", Default, createMask(mem.dataType))
      )
    )
    BundleType(
      (mem.readers.map(Field(_, Flip, rType))) ++
        (mem.writers.map(Field(_, Flip, wType))) ++
        (mem.readwriters.map(Field(_, Flip, rwType)))
    )
  }

  def memPortField(s: DefMemory, p: String, f: String): WSubField = {
    val mem = WRef(s.name, memType(s), MemKind, UnknownFlow)
    val t1 = field_type(mem.tpe, p)
    val t2 = field_type(t1, f)
    WSubField(WSubField(mem, p, t1, UnknownFlow), f, t2, UnknownFlow)
  }
}
