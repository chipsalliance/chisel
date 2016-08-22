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

import com.typesafe.scalalogging.LazyLogging

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.PrimOps._

object seqCat {
  def apply(args: Seq[Expression]): Expression = args.length match {
    case 0 => error("Empty Seq passed to seqcat")
    case 1 => args(0)
    case 2 => DoPrim(PrimOps.Cat, args, Seq.empty[BigInt], UIntType(UnknownWidth))
    case _ => {
      val seqs = args.splitAt(args.length/2)
      DoPrim(PrimOps.Cat, Seq(seqCat(seqs._1),seqCat(seqs._2)), Seq.empty[BigInt], UIntType(UnknownWidth))
    }
  }
}

object toBits {
  def apply(e: Expression): Expression = e match {
    case ex: WRef => hiercat(ex,ex.tpe)
    case ex: WSubField => hiercat(ex,ex.tpe)
    case ex: WSubIndex => hiercat(ex,ex.tpe)
    case t => error("Invalid operand expression for toBits!")
  }
  def hiercat(e: Expression, dt: Type): Expression = dt match {
    case t:VectorType => seqCat((0 until t.size).reverse.map(i => hiercat(WSubIndex(e, i, t.tpe, UNKNOWNGENDER),t.tpe)))
    case t:BundleType => seqCat(t.fields.map(f => hiercat(WSubField(e, f.name, f.tpe, UNKNOWNGENDER), f.tpe)))
    case t:GroundType => e
    case t => error("Unknown type encountered in toBits!")
  }
}

object toBitMask {
  def apply(e: Expression, dataType: Type): Expression = e match {
    case ex: WRef => hiermask(ex,ex.tpe,dataType)
    case ex: WSubField => hiermask(ex,ex.tpe,dataType)
    case ex: WSubIndex => hiermask(ex,ex.tpe,dataType)
    case t => error("Invalid operand expression for toBits!")
  }
  def hiermask(e: Expression, maskType: Type, dataType: Type): Expression = (maskType, dataType) match {
    case (mt:VectorType, dt:VectorType) => seqCat((0 until mt.size).reverse.map(i => hiermask(WSubIndex(e, i, mt.tpe, UNKNOWNGENDER), mt.tpe, dt.tpe)))
    case (mt:BundleType, dt:BundleType) => seqCat((mt.fields zip dt.fields).map{ case (mf,df) =>
      hiermask(WSubField(e, mf.name, mf.tpe, UNKNOWNGENDER), mf.tpe, df.tpe) })
    case (mt:UIntType, dt:GroundType) => seqCat(List.fill(bitWidth(dt).intValue)(e))
    case (mt, dt) => error("Invalid type for mask component!")
  }
}

object bitWidth {
  def apply(dt: Type): BigInt = widthOf(dt)
  def widthOf(dt: Type): BigInt = dt match {
    case t:VectorType => t.size * bitWidth(t.tpe)
    case t:BundleType => t.fields.map(f => bitWidth(f.tpe)).foldLeft(BigInt(0))(_+_)
    case UIntType(IntWidth(width)) => width
    case SIntType(IntWidth(width)) => width
    case t => error("Unknown type encountered in bitWidth!")
  }
}

object fromBits {
  def apply(lhs: Expression, rhs: Expression): Statement = {
    val fbits = lhs match {
      case ex: WRef => getPart(ex, ex.tpe, rhs, 0)
      case ex: WSubField => getPart(ex, ex.tpe, rhs, 0)
      case ex: WSubIndex => getPart(ex, ex.tpe, rhs, 0)
      case t => error("Invalid LHS expression for fromBits!")
    }
    Block(fbits._2)
  }
  def getPartGround(lhs: Expression, lhst: Type, rhs: Expression, offset: BigInt): (BigInt, Seq[Statement]) = {
    val intWidth = bitWidth(lhst)
    val sel = DoPrim(PrimOps.Bits, Seq(rhs), Seq(offset+intWidth-1,offset), UnknownType)
    (offset + intWidth, Seq(Connect(NoInfo,lhs,sel)))
  }
  def getPart(lhs: Expression, lhst: Type, rhs: Expression, offset: BigInt): (BigInt, Seq[Statement]) = {
    lhst match {
      case t:VectorType => {
        var currentOffset = offset
        var stmts = Seq.empty[Statement]
        for (i <- (0 until t.size)) {
          val (tmpOffset, substmts) = getPart(WSubIndex(lhs, i, t.tpe, UNKNOWNGENDER), t.tpe, rhs, currentOffset)
          stmts = stmts ++ substmts
          currentOffset = tmpOffset
        }
        (currentOffset, stmts)
      }
      case t:BundleType => {
        var currentOffset = offset
        var stmts = Seq.empty[Statement]
        for (f <- t.fields.reverse) {
          val (tmpOffset, substmts) = getPart(WSubField(lhs, f.name, f.tpe, UNKNOWNGENDER), f.tpe, rhs, currentOffset)
          stmts = stmts ++ substmts
          currentOffset = tmpOffset
        }
        (currentOffset, stmts)
      }
      case t:GroundType => getPartGround(lhs, t, rhs, offset)
      case t => error("Unknown type encountered in fromBits!")
    }
  }
}

object MemPortUtils {

  import AnalysisUtils._

  def flattenType(t: Type) = UIntType(IntWidth(bitWidth(t)))

  def defaultPortSeq(mem: DefMemory) = Seq(
    Field("addr", Default, UIntType(IntWidth(ceil_log2(mem.depth)))),
    Field("en", Default, UIntType(IntWidth(1))),
    Field("clk", Default, ClockType)
  )

  def rPortToBundle(mem: DefMemory) = BundleType(defaultPortSeq(mem) :+ Field("data", Flip, mem.dataType))
  def rPortToFlattenBundle(mem: DefMemory) = BundleType(defaultPortSeq(mem) :+ Field("data", Flip, flattenType(mem.dataType)))

  def wPortToBundle(mem: DefMemory) = {
    val defaultSeq = defaultPortSeq(mem) :+ Field("data", Default, mem.dataType)
    BundleType(
      if (containsInfo(mem.info,"maskGran")) defaultSeq :+ Field("mask", Default, create_mask(mem.dataType))
      else defaultSeq
    )
  }
  def wPortToFlattenBundle(mem: DefMemory) = {
    val defaultSeq = defaultPortSeq(mem) :+ Field("data", Default, flattenType(mem.dataType))
    BundleType(
      if (containsInfo(mem.info,"maskGran")) defaultSeq :+ Field("mask", Default, flattenType(create_mask(mem.dataType)))
      else defaultSeq
    )  
  }

  def rwPortToBundle(mem: DefMemory) ={
    val defaultSeq = defaultPortSeq(mem) ++ Seq(
      Field("wmode", Default, UIntType(IntWidth(1))),
      Field("wdata", Default, mem.dataType),
      Field("rdata", Flip, mem.dataType)
    )
    BundleType(
      if (containsInfo(mem.info,"maskGran")) defaultSeq :+ Field("wmask", Default, create_mask(mem.dataType))
      else defaultSeq
    )
  }
  def rwPortToFlattenBundle(mem: DefMemory) ={
    val defaultSeq = defaultPortSeq(mem) ++ Seq(
      Field("wmode", Default, UIntType(IntWidth(1))),
      Field("wdata", Default, flattenType(mem.dataType)),
      Field("rdata", Flip, flattenType(mem.dataType))
    )
    BundleType(
      if (containsInfo(mem.info,"maskGran")) defaultSeq :+ Field("wmask", Default, flattenType(create_mask(mem.dataType)))
      else defaultSeq
    )
  }

  def memToBundle(s: DefMemory) = BundleType(
    s.readers.map(p => Field(p, Default, rPortToBundle(s))) ++
      s.writers.map(p => Field(p, Default, wPortToBundle(s))) ++
      s.readwriters.map(p => Field(p, Default, rwPortToBundle(s))))
  def memToFlattenBundle(s: DefMemory) = BundleType(
    s.readers.map(p => Field(p, Default, rPortToFlattenBundle(s))) ++
      s.writers.map(p => Field(p, Default, wPortToFlattenBundle(s))) ++
      s.readwriters.map(p => Field(p, Default, rwPortToFlattenBundle(s))))
}
