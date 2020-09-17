// SPDX-License-Identifier: Apache-2.0

package firrtl.fuzzer

import firrtl.ir._
import firrtl.{PrimOps, Utils}

import scala.language.higherKinds

/** A generator that generates expressions of a certain type for a given IR type
  */
trait ExprGen[E <: Expression] { self =>
  type Width = BigInt

  /** The name of this generator, used for logging
    */
  def name: String

  /** A [[StateGen]] that produces a one width UInt [[firrtl.ir.Expression Expression]]
    */
  def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, E]]

  /** Takes a width and returns a [[StateGen]] that produces a UInt [[firrtl.ir.Expression Expression]] with the given width
    *
    * The input width will be greater than 1 and less than or equal to the
    * maximum width allowed specified by the input state
    */
  def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, E]]

  /** A [[StateGen]] that produces a one width SInt [[firrtl.ir.Expression Expression]]
    */
  def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, E]]

  /** Takes a width and returns a [[StateGen]] that produces a SInt [[firrtl.ir.Expression Expression]] with the given width
    *
    * The input width will be greater than 1 and less than or equal to the
    * maximum width allowed by the input state
    */
  def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, E]]

  /** Returns a copy of this [[ExprGen]] that throws better exceptions
    *
    * Wraps this [[ExprGen]]'s functions with a try-catch that will wrap
    * exceptions and record the name of this generator in an
    * [[ExprGen.TraceException]].
    */
  def withTrace: ExprGen[E] = new ExprGen[E] {
    import GenMonad.syntax._

    def name = self.name

    private def wrap[S: ExprState, G[_]: GenMonad](
      name: String,
      tpe: Type,
      stateGen: StateGen[S, G, E]): StateGen[S, G, E] = {
      StateGen { (s: S) =>
        GenMonad[G].choose(0, 1).map ( _ =>
          try {
            GenMonad[G].generate(stateGen.run(s))
          } catch {
            case e: ExprGen.TraceException if e.trace.size < 10 =>
              throw e.copy(trace = s"$name: ${tpe.serialize}" +: e.trace)
            case e: IllegalArgumentException =>
              throw ExprGen.TraceException(Seq(s"$name: ${tpe.serialize}"), e)
          }
        )
      }
    }

    /** Identical to self.boolUIntGen, but will wrap exceptions in a [[ExprGen.TraceException]]
      */
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, E]] = {
      self.boolUIntGen.map(wrap(self.name, Utils.BoolType, _))
    }

    /** Identical to self.uintGen, but will wrap exceptions in a [[ExprGen.TraceException]]
      */
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, E]] = {
      self.uintGen.map(fn => (width: Width) => wrap(self.name, UIntType(IntWidth(width)), fn(width)))
    }

    /** Identical to self.boolSIntGen, but will wrap exceptions in a [[ExprGen.TraceException]]
      */
    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, E]] = {
      self.boolSIntGen.map(wrap(self.name, SIntType(IntWidth(1)), _))
    }

    /** Identical to self.sintGen, but will wrap exceptions in a [[ExprGen.TraceException]]
      */
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, E]] = {
      self.sintGen.map(fn => (width: Width) => wrap(self.name, SIntType(IntWidth(width)), fn(width)))
    }
  }
}

/** An Expression Generator that generates [[firrtl.ir.DoPrim DoPrim]]s of the given operator
  */
abstract class DoPrimGen(val primOp: PrimOp) extends ExprGen[DoPrim] {
  def name = primOp.serialize
}

object ExprGen {
  type Width = BigInt

  import GenMonad.syntax._

  private def printStack(e: Throwable): String = {
    val sw = new java.io.StringWriter()
    val pw = new java.io.PrintWriter(sw)
    e.printStackTrace(pw)
    pw.flush()
    sw.toString
  }

  case class TraceException(trace: Seq[String], cause: Throwable) extends Exception(
    s"cause: ${printStack(cause)}\ntrace:\n${trace.reverse.mkString("\n")}\n"
  )

  def genWidth[G[_]: GenMonad](min: Int, max: Int): G[IntWidth] = GenMonad[G].choose(min, max).map(IntWidth(_))
  def genWidthMax[G[_]: GenMonad](max: Int): G[IntWidth] = genWidth(1, max)

  private def makeBinDoPrimStateGen[S: ExprState, G[_]: GenMonad](
    primOp: PrimOp,
    typeGen: Int => G[(Type, Type, Type)]): StateGen[S, G, DoPrim] = {
    for {
      t <- StateGen.inspectG((s: S) => typeGen(ExprState[S].maxWidth(s)))
      (tpe1, tpe2, exprTpe) = t
      expr1 <- ExprState[S].exprGen(tpe1)
      expr2 <- ExprState[S].exprGen(tpe2)
    } yield {
      DoPrim(primOp, Seq(expr1, expr2), Seq.empty, exprTpe)
    }
  }

  private def makeUnaryDoPrimStateGen[S: ExprState, G[_]: GenMonad](
    primOp: PrimOp,
    typeGen: Int => G[(Type, Type)]): StateGen[S, G, DoPrim] = {
    for {
      p <- StateGen.inspectG((s: S) => typeGen(ExprState[S].maxWidth(s)))
      (tpe1, exprTpe) = p
      expr1 <- ExprState[S].exprGen(tpe1)
    } yield {
      DoPrim(primOp, Seq(expr1), Seq.empty, exprTpe)
    }
  }

  private def log2Floor(i: Int): Int = {
    if (i > 0 && ((i & (i - 1)) == 0)) {
      BigInt(i - 1).bitLength
    } else {
      BigInt(i - 1).bitLength - 1
    }
  }

  def inspectMinWidth[S: ExprState, G[_]: GenMonad](min: Int): StateGen[S, G, IntWidth] = {
    StateGen.inspectG { s =>
      val maxWidth = ExprState[S].maxWidth(s)
      GenMonad[G].choose(min, maxWidth).map(IntWidth(_))
    }
  }


  object ReferenceGen extends ExprGen[Reference] {
    def name = "Ref"
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, Reference]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, Reference]] = Some { width =>
      referenceGenImp(UIntType(IntWidth(width)))
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, Reference]]  = sintGen.map(_(1))
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, Reference]] = Some { width =>
      referenceGenImp(SIntType(IntWidth(width)))
    }

    private def referenceGenImp[S: ExprState, G[_]: GenMonad](tpe: Type): StateGen[S, G, Reference] = {
      for {
        // don't bother randomizing name, because it usually does not help with coverage
        tryName <- StateGen.pure("ref")
        ref <- ExprState[S].withRef(Reference(tryName, tpe))
      } yield {
        ref
      }
    }
  }

  object LiteralGen extends ExprGen[Literal] {
    def name = "Literal"
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, Literal]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, Literal]] = Some { width =>
      uintLiteralGenImp(width)
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, Literal]] = sintGen.map(_(1))
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, Literal]] = Some { width =>
      sintLiteralGenImp(width)
    }

    private def uintLiteralGenImp[S: ExprState, G[_]: GenMonad](width: Width): StateGen[S, G, Literal] = {
      val maxValue = if (width < BigInt(31)) {
        (1 << width.toInt) - 1
      } else {
        Int.MaxValue
      }
      StateGen.liftG(for {
        value <- GenMonad[G].choose(0, maxValue)
      } yield {
        UIntLiteral(value, IntWidth(width))
      })
    }

    private def sintLiteralGenImp[S: ExprState, G[_]: GenMonad](width: Width): StateGen[S, G, Literal] = {
      StateGen.liftG(
        if (width <= BigInt(32)) {
          GenMonad[G].choose(-(1 << (width.toInt - 1)), (1 << (width.toInt - 1)) - 1).map { value =>
            SIntLiteral(value, IntWidth(width))
          }
        } else {
          GenMonad[G].choose(Int.MinValue, Int.MaxValue).map { value =>
            SIntLiteral(value, IntWidth(width))
          }
        }
      )
    }
  }

  abstract class AddSubDoPrimGen(isAdd: Boolean) extends DoPrimGen(if (isAdd) PrimOps.Add else PrimOps.Sub) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { width =>
        def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
          randWidth <- GenMonad[G].choose(1, width.toInt - 1)
          flip <- GenMonad.bool
        } yield {
          if (flip) {
            (UIntType(IntWidth(randWidth)), UIntType(IntWidth(width.toInt - 1)), UIntType(IntWidth(width)))
          } else {
            (UIntType(IntWidth(width.toInt - 1)), UIntType(IntWidth(randWidth)), UIntType(IntWidth(width)))
          }
        }
        makeBinDoPrimStateGen(primOp, typeGen)
      }
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { width =>
        def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
          randWidth <- GenMonad[G].choose(1, width.toInt - 1)
          flip <- GenMonad.bool
        } yield {
          if (flip) {
            (SIntType(IntWidth(randWidth)), SIntType(IntWidth(width.toInt - 1)), SIntType(IntWidth(width)))
          } else {
            (SIntType(IntWidth(width.toInt - 1)), SIntType(IntWidth(randWidth)), SIntType(IntWidth(width)))
          }
        }
        makeBinDoPrimStateGen(primOp, typeGen)
      }
    }
  }

  object AddDoPrimGen extends AddSubDoPrimGen(isAdd = true)
  object SubDoPrimGen extends AddSubDoPrimGen(isAdd = false)

  object MulDoPrimGen extends DoPrimGen(PrimOps.Mul) {
    private def imp[S: ExprState, G[_]: GenMonad](isUInt: Boolean): Option[Width => StateGen[S, G, DoPrim]] = {
      val tpe = if (isUInt) UIntType(_) else SIntType(_)
      Some { totalWidth =>
        def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
          width1 <- GenMonad[G].choose(1, totalWidth.toInt - 1)
          flip <- GenMonad.bool
        } yield {
          val t1 = tpe(IntWidth(math.max(totalWidth.toInt - width1, 0)))
          val t2 = tpe(IntWidth(width1))
          val t3 = tpe(IntWidth(totalWidth))
          if (flip) (t2, t1, t3) else (t1, t2, t3)
        }
        makeBinDoPrimStateGen(primOp, typeGen)
      }
    }

    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = true)

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = false)
  }

  object DivDoPrimGen extends DoPrimGen(PrimOps.Div) {
    private def imp[S: ExprState, G[_]: GenMonad](isUInt: Boolean): Option[Width => StateGen[S, G, DoPrim]] = {
      val tpe = if (isUInt) UIntType(_) else SIntType(_)
      Some { resultWidth =>
        val numWidth = if (isUInt) resultWidth.toInt else (resultWidth.toInt - 1)
        def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
          denWidth <- genWidthMax(maxWidth)
        } yield {
          (tpe(IntWidth(numWidth)), tpe(denWidth), tpe(IntWidth(resultWidth)))
        }
        makeBinDoPrimStateGen(primOp, typeGen)
      }
    }

    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = true)

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = false)
  }

  object RemDoPrimGen extends DoPrimGen(PrimOps.Rem) {
    private def imp[S: ExprState, G[_]: GenMonad](isUInt: Boolean): Option[Width => StateGen[S, G, DoPrim]] = {
      val tpe = if (isUInt) UIntType(_) else SIntType(_)
      Some { resultWidth =>
        def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
          argWidth <- genWidth(resultWidth.toInt, maxWidth)
          flip <- GenMonad.bool
        } yield {
          val t1 = UIntType(argWidth)
          val t2 = UIntType(IntWidth(resultWidth))
          val t3 = t2
          if (flip) (t2, t1, t3) else (t1, t2, t3)
        }
        makeBinDoPrimStateGen(primOp, typeGen)
      }
    }

    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = true)

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = sintGen.map(_(1))
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = false)
  }

  abstract class CmpDoPrimGen(primOp: PrimOp) extends DoPrimGen(primOp) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = {
      def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
        width1 <- genWidthMax(maxWidth)
        width2 <- genWidthMax(maxWidth)
        tpe <- GenMonad[G].oneOf(UIntType(_), SIntType(_))
      } yield {
        (tpe(width1), tpe(width2), Utils.BoolType)
      }
      Some(makeBinDoPrimStateGen(primOp, typeGen))
    }
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None
  }
  object LtDoPrimGen extends CmpDoPrimGen(PrimOps.Lt)
  object LeqDoPrimGen extends CmpDoPrimGen(PrimOps.Leq)
  object GtDoPrimGen extends CmpDoPrimGen(PrimOps.Gt)
  object GeqDoPrimGen extends CmpDoPrimGen(PrimOps.Geq)
  object EqDoPrimGen extends CmpDoPrimGen(PrimOps.Eq)
  object NeqDoPrimGen extends CmpDoPrimGen(PrimOps.Neq)

  object PadDoPrimGen extends DoPrimGen(PrimOps.Pad) {
    private def imp[S: ExprState, G[_]: GenMonad](isUInt: Boolean): Option[Width => StateGen[S, G, DoPrim]] = {
      val tpe = if (isUInt) UIntType(_) else SIntType(_)
      Some { resultWidth =>
        for {
          width1 <- StateGen.liftG(genWidthMax(resultWidth.toInt))
          flip <- StateGen.liftG(GenMonad.bool)
          expr <- ExprState[S].exprGen(tpe(if (flip) IntWidth(resultWidth) else width1))
        } yield {
          DoPrim(primOp, Seq(expr), Seq(if (flip) width1.width else resultWidth), tpe(IntWidth(resultWidth)))
        }
      }
    }

    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = true)

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = sintGen.map(_(1))
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = false)
  }

  object ShlDoPrimGen extends DoPrimGen(PrimOps.Shl) {
    private def imp[S: ExprState, G[_]: GenMonad](isUInt: Boolean): Option[Width => StateGen[S, G, DoPrim]] = {
      val tpe = if (isUInt) UIntType(_) else SIntType(_)
      Some { totalWidth =>
        for {
          width1 <- StateGen.liftG(genWidthMax(totalWidth.toInt))
          expr <- ExprState[S].exprGen(tpe(width1))
        } yield {
          val width2 = totalWidth.toInt - width1.width.toInt
          DoPrim(primOp, Seq(expr), Seq(width2), tpe(IntWidth(totalWidth)))
        }
      }
    }

    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = true)

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = false)
  }

  object ShrDoPrimGen extends DoPrimGen(PrimOps.Shr) {
    private def imp[S: ExprState, G[_]: GenMonad](isUInt: Boolean): Option[Width => StateGen[S, G, DoPrim]] = {
      val tpe = if (isUInt) UIntType(_) else SIntType(_)
      Some { minWidth =>
        for {
          exprWidth <- inspectMinWidth(minWidth.toInt)
          expr <- ExprState[S].exprGen(tpe(exprWidth))
          shamt <- StateGen.liftG(if (minWidth == BigInt(1)) {
            GenMonad[G].choose(exprWidth.width.toInt - minWidth.toInt, Int.MaxValue)
          } else {
            GenMonad[G].const(exprWidth.width.toInt - minWidth.toInt)
          })
        } yield {
          DoPrim(primOp, Seq(expr), Seq(shamt), tpe(IntWidth(minWidth)))
        }
      }
    }

    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = true)

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = sintGen.map(_(1))
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = false)
  }

  object DshlDoPrimGen extends DoPrimGen(PrimOps.Dshl) {
    private def imp[S: ExprState, G[_]: GenMonad](isUInt: Boolean): Option[Width => StateGen[S, G, DoPrim]] = {
      val tpe = if (isUInt) UIntType(_) else SIntType(_)
      Some { totalWidth =>
        val shWidthMax = log2Floor(totalWidth.toInt)
        def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
          shWidth <- genWidthMax(shWidthMax)
        } yield {
          val w1 = totalWidth.toInt - ((1 << shWidth.width.toInt) - 1)
          // println(s"TOTALWIDTH: $totalWidth, SHWIDHTMAX: ${shWidthMax}, SHWIDTH: $shWidth, ARGWITH: $w1")
          (tpe(IntWidth(w1)), UIntType(shWidth), tpe(IntWidth(totalWidth)))
        }
        makeBinDoPrimStateGen(primOp, typeGen)
      }
    }

    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = true)

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = false)
  }

  object DshrDoPrimGen extends DoPrimGen(PrimOps.Dshr) {
    private def imp[S: ExprState, G[_]: GenMonad](isUInt: Boolean): Option[Width => StateGen[S, G, DoPrim]] = {
      val tpe = if (isUInt) UIntType(_) else SIntType(_)
      Some { width =>
        def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
          w2 <- genWidthMax(maxWidth)
        } yield {
          (tpe(IntWidth(width)), tpe(w2), tpe(IntWidth(width)))
        }
        makeBinDoPrimStateGen(primOp, typeGen)
      }
    }

    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = true)

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = sintGen.map(_(1))
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = imp(isUInt = false)
  }

  object CvtDoPrimGen extends DoPrimGen(PrimOps.Cvt) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = {
      Some {
        def typeGen(maxWidth: Int): G[(Type, Type)] = {
          val resultType = SIntType(IntWidth(1))
          GenMonad[G].const(resultType -> resultType)
        }
        makeUnaryDoPrimStateGen(primOp, typeGen)
      }
    }
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { width =>
        def typeGen(maxWidth: Int): G[(Type, Type)] = for {
          isUInt <- GenMonad[G].oneOf(true, false)
        } yield {
          val resultType = SIntType(IntWidth(width))
          if (isUInt) {
            UIntType(IntWidth(width.toInt - 1)) -> resultType
          } else {
            resultType -> resultType
          }
        }
        makeUnaryDoPrimStateGen(primOp, typeGen)
      }
    }
  }

  object NegDoPrimGen extends DoPrimGen(PrimOps.Neg) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { width =>
        def typeGen(maxWidth: Int): G[(Type, Type)] = for {
          isUInt <- GenMonad[G].oneOf(true, false)
        } yield {
          val resultType = SIntType(IntWidth(width))
          if (isUInt) {
            UIntType(IntWidth(width.toInt - 1)) -> resultType
          } else {
            SIntType(IntWidth(width.toInt - 1)) -> resultType
          }
        }
        makeUnaryDoPrimStateGen(primOp, typeGen)
      }
    }
  }

  object NotDoPrimGen extends DoPrimGen(PrimOps.Not) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { width =>
        def typeGen(maxWidth: Int): G[(Type, Type)] = for {
          isUInt <- GenMonad[G].oneOf(true, false)
        } yield {
          val resultType = UIntType(IntWidth(width))
          if (isUInt) {
            resultType -> resultType
          } else {
            SIntType(IntWidth(width)) -> resultType
          }
        }
        makeUnaryDoPrimStateGen(primOp, typeGen)
      }
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None
  }

  abstract class BitwiseDoPrimGen(primOp: PrimOp) extends DoPrimGen(primOp) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { width =>
        def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
          width1 <- genWidthMax(width.toInt)
          flip <- GenMonad.bool
          tpe <- GenMonad[G].oneOf(UIntType(_), SIntType(_))
        } yield {
          val t1 = tpe(width1)
          val t2 = tpe(IntWidth(width))
          val t3 = UIntType(IntWidth(width))
          if (flip) (t2, t1, t3) else (t1, t2, t3)
        }
        makeBinDoPrimStateGen(primOp, typeGen)
      }
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None
  }
  object AndDoPrimGen extends BitwiseDoPrimGen(PrimOps.And)
  object OrDoPrimGen extends BitwiseDoPrimGen(PrimOps.Or)
  object XorDoPrimGen extends BitwiseDoPrimGen(PrimOps.Xor)

  abstract class ReductionDoPrimGen(primOp: PrimOp) extends DoPrimGen(primOp) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = {
      Some {
        def typeGen(maxWidth: Int): G[(Type, Type)] = for {
          width1 <- genWidthMax(maxWidth)
          tpeFn <- GenMonad[G].oneOf(UIntType(_), SIntType(_))
        } yield {
          (tpeFn(width1), Utils.BoolType)
        }
        makeUnaryDoPrimStateGen(primOp, typeGen)
      }
    }
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None
  }
  object AndrDoPrimGen extends ReductionDoPrimGen(PrimOps.Andr)
  object OrrDoPrimGen extends ReductionDoPrimGen(PrimOps.Orr)
  object XorrDoPrimGen extends ReductionDoPrimGen(PrimOps.Xorr)

  object CatDoPrimGen extends DoPrimGen(PrimOps.Cat) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { totalWidth =>
        def typeGen(maxWidth: Int): G[(Type, Type, Type)] = for {
          width1 <- GenMonad[G].choose(1, totalWidth.toInt - 1)
          flip <- GenMonad.bool
          tpe <- GenMonad[G].oneOf(UIntType(_), SIntType(_))
        } yield {
          val t1 = tpe(IntWidth(totalWidth.toInt - width1))
          val t2 = tpe(IntWidth(width1))
          val t3 = UIntType(IntWidth(totalWidth))
          if (flip) (t2, t1, t3) else (t1, t2, t3)
        }
        makeBinDoPrimStateGen(primOp, typeGen)
      }
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None
  }

  object BitsDoPrimGen extends DoPrimGen(PrimOps.Bits) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { resultWidth =>
        for {
          argWidth <- inspectMinWidth(resultWidth.toInt)
          lo <- StateGen.liftG(GenMonad[G].choose(0, argWidth.width.toInt - resultWidth.toInt))
          tpe <- StateGen.liftG(GenMonad[G].oneOf(UIntType(_), SIntType(_)))
          arg <- ExprState[S].exprGen(tpe(argWidth))
        } yield {
          DoPrim(primOp, Seq(arg), Seq(lo + resultWidth - 1, lo), UIntType(IntWidth(resultWidth)))
        }
      }
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None
  }

  object HeadDoPrimGen extends DoPrimGen(PrimOps.Head) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { resultWidth =>
        for {
          argWidth <- inspectMinWidth(resultWidth.toInt)
          tpe <- StateGen.liftG(GenMonad[G].oneOf(UIntType(_), SIntType(_)))
          arg <- ExprState[S].exprGen(tpe(argWidth))
        } yield {
          DoPrim(primOp, Seq(arg), Seq(resultWidth), UIntType(IntWidth(resultWidth)))
        }
      }
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None
  }

  object TailDoPrimGen extends DoPrimGen(PrimOps.Tail) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { totalWidth =>
        for {
          maxWidth <- StateGen.inspect((s: S) => ExprState[S].maxWidth(s))
          tailNum <- if (totalWidth < BigInt(maxWidth)) {
            StateGen.liftG[S, G, Int](GenMonad[G].choose(1, maxWidth - totalWidth.toInt))
          } else {
            StateGen.pure[S, G, Int](0)
          }
          tpe <- StateGen.liftG(GenMonad[G].oneOf(UIntType(_), SIntType(_)))
          arg <- ExprState[S].exprGen(tpe(IntWidth(totalWidth + tailNum)))
        } yield {
          DoPrim(primOp, Seq(arg), Seq(tailNum), UIntType(IntWidth(totalWidth)))
        }
      }
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None
  }

  object AsUIntDoPrimGen extends DoPrimGen(PrimOps.AsUInt) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { width =>
        val intWidth = IntWidth(width)
        for {
          tpe <- StateGen.liftG(GenMonad[G].oneOf(UIntType(_), SIntType(_)))
          arg <- ExprState[S].exprGen(tpe(intWidth))
        } yield {
          DoPrim(primOp, Seq(arg), Seq(), UIntType(intWidth))
        }
      }
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None
  }

  object AsSIntDoPrimGen extends DoPrimGen(PrimOps.AsSInt) {
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = None
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = None

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, DoPrim]] = sintGen.map(_(1))
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, DoPrim]] = {
      Some { width =>
        val intWidth = IntWidth(width)
        for {
          tpe <- StateGen.liftG(GenMonad[G].oneOf(UIntType(_), SIntType(_)))
          arg <- ExprState[S].exprGen(tpe(intWidth))
        } yield {
          DoPrim(primOp, Seq(arg), Seq(), SIntType(intWidth))
        }
      }
    }
  }

  object MuxGen extends ExprGen[Mux] {
    def name = "mux"
    private def imp[S: ExprState, G[_]: GenMonad](isUInt: Boolean)(width: BigInt): StateGen[S, G, Mux] = {
      val tpe = if (isUInt) UIntType(_) else SIntType(_)
      // spec states that types must be equivalent, but in practice we allow differing widths
      for {
        cond <- ExprState[S].exprGen(Utils.BoolType)
        flip <- StateGen.liftG(GenMonad.bool)
        expr1 <- ExprState[S].exprGen(tpe(IntWidth(width)))
        width2 <- StateGen.liftG(genWidthMax(width.toInt))
        expr2 <- ExprState[S].exprGen(tpe(width2))
      } yield {
        Mux(cond, expr1, expr2, tpe(IntWidth(width)))
      }
    }
    def boolUIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, Mux]] = uintGen.map(_(1))
    def uintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, Mux]] = {
      Some { imp(isUInt = true) }
    }

    def boolSIntGen[S: ExprState, G[_]: GenMonad]: Option[StateGen[S, G, Mux]] = sintGen.map(_(1))
    def sintGen[S: ExprState, G[_]: GenMonad]: Option[Width => StateGen[S, G, Mux]] = {
      Some { imp(isUInt = false) }
    }
  }
}
