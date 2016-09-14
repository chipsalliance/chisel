/*
 Copyright (c) 2014 - 2016 The Regents of the University of
 California (Regents). All Rights Reserved.  Redistribution and use in
 source and binary forms, with or without modification, are permitted
 provided that the following conditions are met:
 * Redistributions of source code must retain the above
 copyright notice, this list of conditions and the following
 two paragraphs of disclaimer.
 * Redistributions in binary form must reproduce the above
 copyright notice, this list of conditions and the following
 two paragraphs of disclaimer in the documentation and/or other materials
 provided with the distribution.
 * Neither the name of the Regents nor the names of its contributors
 may be used to endorse or promote products derived from this
 software without specific prior written permission.
 IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
 SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
 ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
 REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
 LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
 ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
 TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
 MODIFICATIONS.
 */

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.PrimOps._

object seqCat {
  def apply(args: Seq[Expression]): Expression = args.length match {
    case 0 => error("Empty Seq passed to seqcat")
    case 1 => args(0)
    case 2 => DoPrim(PrimOps.Cat, args, Nil, UIntType(UnknownWidth))
    case _ =>
      val (high, low) = args splitAt (args.length / 2)
      DoPrim(PrimOps.Cat, Seq(seqCat(high), seqCat(low)), Nil, UIntType(UnknownWidth))
  }
}

object toBits {
  def apply(e: Expression): Expression = e match {
    case ex @ (_: WRef | _: WSubField | _: WSubIndex) => hiercat(ex, ex.tpe)
    case t => error("Invalid operand expression for toBits!")
  }
  private def hiercat(e: Expression, dt: Type): Expression = dt match {
    case t: VectorType => seqCat((0 until t.size) map (i =>
      hiercat(WSubIndex(e, i, t.tpe, UNKNOWNGENDER),t.tpe)))
    case t: BundleType => seqCat(t.fields map (f =>
      hiercat(WSubField(e, f.name, f.tpe, UNKNOWNGENDER), f.tpe)))
    case t: GroundType => e
    case t => error("Unknown type encountered in toBits!")
  }
}

// TODO: make easier to understand
object toBitMask {
  def apply(e: Expression, dataType: Type): Expression = e match {
    case ex @ (_: WRef | _: WSubField | _: WSubIndex) => hiermask(ex, ex.tpe, dataType)
    case t => error("Invalid operand expression for toBits!")
  }
  private def hiermask(e: Expression, maskType: Type, dataType: Type): Expression =
    (maskType, dataType) match {
      case (mt: VectorType, dt: VectorType) =>
        seqCat((0 until mt.size).reverse map { i =>
          hiermask(WSubIndex(e, i, mt.tpe, UNKNOWNGENDER), mt.tpe, dt.tpe)
        })
      case (mt: BundleType, dt: BundleType) =>
        seqCat((mt.fields zip dt.fields) map { case (mf, df) =>
          hiermask(WSubField(e, mf.name, mf.tpe, UNKNOWNGENDER), mf.tpe, df.tpe)
        })
      case (mt: UIntType, dt: GroundType) =>
        seqCat(List.fill(bitWidth(dt).intValue)(e))
      case (mt, dt) => error("Invalid type for mask component!")
    }
}

object getWidth {
  def apply(t: Type): Width = t match {
    case t: GroundType => t.width
    case _ => error("No width!")
  }
  def apply(e: Expression): Width = apply(e.tpe)
}

object bitWidth {
  def apply(dt: Type): BigInt = widthOf(dt)
  private def widthOf(dt: Type): BigInt = dt match {
    case t: VectorType => t.size * bitWidth(t.tpe)
    case t: BundleType => t.fields.map(f => bitWidth(f.tpe)).foldLeft(BigInt(0))(_+_)
    case GroundType(IntWidth(width)) => width
    case t => error("Unknown type encountered in bitWidth!")
  }
}

object fromBits {
  def apply(lhs: Expression, rhs: Expression): Statement = {
    val fbits = lhs match {
      case ex @ (_: WRef | _: WSubField | _: WSubIndex) => getPart(ex, ex.tpe, rhs, 0)
      case _ => error("Invalid LHS expression for fromBits!")
    }
    Block(fbits._2)
  }
  private def getPartGround(lhs: Expression,
                            lhst: Type,
                            rhs: Expression,
                            offset: BigInt): (BigInt, Seq[Statement]) = {
    val intWidth = bitWidth(lhst)
    val sel = DoPrim(PrimOps.Bits, Seq(rhs), Seq(offset + intWidth - 1, offset), UnknownType)
    (offset + intWidth, Seq(Connect(NoInfo, lhs, sel)))
  }
  private def getPart(lhs: Expression,
                      lhst: Type,
                      rhs: Expression,
                      offset: BigInt): (BigInt, Seq[Statement]) =
    lhst match {
      case t: VectorType => (0 until t.size foldRight (offset, Seq[Statement]())) {
        case (i, (curOffset, stmts)) =>
          val subidx = WSubIndex(lhs, i, t.tpe, UNKNOWNGENDER)
          val (tmpOffset, substmts) = getPart(subidx, t.tpe, rhs, curOffset)
          (tmpOffset, stmts ++ substmts)
      }
      case t: BundleType => (t.fields foldRight (offset, Seq[Statement]())) {
        case (f, (curOffset, stmts)) =>
          val subfield = WSubField(lhs, f.name, f.tpe, UNKNOWNGENDER)
          val (tmpOffset, substmts) = getPart(subfield, f.tpe, rhs, curOffset)
          (tmpOffset, stmts ++ substmts)
      }
      case t: GroundType => getPartGround(lhs, t, rhs, offset)
      case t => error("Unknown type encountered in fromBits!")
    }
}

object createMask {
  def apply(dt: Type): Type = dt match {
    case t: VectorType => VectorType(apply(t.tpe), t.size)
    case t: BundleType => BundleType(t.fields map (f => f copy (tpe=apply(f.tpe))))
    case t: UIntType => BoolType
    case t: SIntType => BoolType
  }
}

object MemPortUtils {

  import AnalysisUtils._

  def flattenType(t: Type) = UIntType(IntWidth(bitWidth(t)))

  def defaultPortSeq(mem: DefMemory) = Seq(
    Field("addr", Default, UIntType(IntWidth(ceil_log2(mem.depth) max 1))),
    Field("en", Default, BoolType),
    Field("clk", Default, ClockType)
  )

  def getFillWMask(mem: DefMemory) =
    getInfo(mem.info, "maskGran") match {
      case None => false
      case Some(maskGran) => maskGran == 1
    }

  def rPortToBundle(mem: DefMemory) = BundleType(
    defaultPortSeq(mem) :+ Field("data", Flip, mem.dataType))
  def rPortToFlattenBundle(mem: DefMemory) = BundleType(
    defaultPortSeq(mem) :+ Field("data", Flip, flattenType(mem.dataType)))

  def wPortToBundle(mem: DefMemory) = BundleType(
    (defaultPortSeq(mem) :+ Field("data", Default, mem.dataType)) ++
    (if (!containsInfo(mem.info, "maskGran")) Nil
     else Seq(Field("mask", Default, createMask(mem.dataType))))
  )
  def wPortToFlattenBundle(mem: DefMemory) = BundleType(
    (defaultPortSeq(mem) :+ Field("data", Default, flattenType(mem.dataType))) ++
    (if (!containsInfo(mem.info, "maskGran")) Nil
     else if (getFillWMask(mem)) Seq(Field("mask", Default, flattenType(mem.dataType)))
     else Seq(Field("mask", Default, flattenType(createMask(mem.dataType)))))
  )
  // TODO: Don't use createMask???

  def rwPortToBundle(mem: DefMemory) = BundleType(
    defaultPortSeq(mem) ++ Seq(
      Field("wmode", Default, BoolType),
      Field("wdata", Default, mem.dataType),
      Field("rdata", Flip, mem.dataType)
    ) ++ (if (!containsInfo(mem.info, "maskGran")) Nil
     else Seq(Field("wmask", Default, createMask(mem.dataType)))
    )
  )

  def rwPortToFlattenBundle(mem: DefMemory) = BundleType(
    defaultPortSeq(mem) ++ Seq(
      Field("wmode", Default, BoolType),
      Field("wdata", Default, flattenType(mem.dataType)),
      Field("rdata", Flip, flattenType(mem.dataType))
    ) ++ (if (!containsInfo(mem.info, "maskGran")) Nil
     else if (getFillWMask(mem)) Seq(Field("wmask", Default, flattenType(mem.dataType)))
     else Seq(Field("wmask", Default, flattenType(createMask(mem.dataType))))
    )  
  )

  def memToBundle(s: DefMemory) = BundleType(
    s.readers.map(Field(_, Flip, rPortToBundle(s))) ++
    s.writers.map(Field(_, Flip, wPortToBundle(s))) ++
    s.readwriters.map(Field(_, Flip, rwPortToBundle(s))))
  
  def memToFlattenBundle(s: DefMemory) = BundleType(
    s.readers.map(Field(_, Flip, rPortToFlattenBundle(s))) ++
    s.writers.map(Field(_, Flip, wPortToFlattenBundle(s))) ++
    s.readwriters.map(Field(_, Flip, rwPortToFlattenBundle(s))))

  // Todo: merge it with memToBundle
  def memType(mem: DefMemory) = {
    val rType = rPortToBundle(mem)
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

  def memPortField(s: DefMemory, p: String, f: String) = {
    val mem = WRef(s.name, memType(s), MemKind, UNKNOWNGENDER)
    val t1 = field_type(mem.tpe, p)
    val t2 = field_type(t1, f)
    WSubField(WSubField(mem, p, t1, UNKNOWNGENDER), f, t2, UNKNOWNGENDER)
  }
}
