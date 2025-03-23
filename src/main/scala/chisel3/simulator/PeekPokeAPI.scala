package chisel3.simulator

import svsim._
import chisel3._

import chisel3.reflect.DataMirror
import chisel3.experimental.{SourceInfo, SourceLine}
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.VecLiterals._
import chisel3.internal.ExceptionHelpers
import firrtl.options.StageUtils.dramaticMessage
import scala.util.control.NoStackTrace

object PeekPokeAPI extends PeekPokeAPI

trait PeekPokeAPI {
  case class FailedExpectationException[T](observed: T, expected: T, message: String)
      extends RuntimeException(
        dramaticMessage(
          header = Some("Failed Expectation"),
          body = s"""|Observed value: '$observed'
                     |Expected value: '$expected'
                     |$message""".stripMargin
        )
      )
      with NoStackTrace
  object FailedExpectationException {
    def apply[T](
      observed:     T,
      expected:     T,
      message:      String,
      sourceInfo:   SourceInfo,
      extraContext: Seq[String]
    ): FailedExpectationException[T] = {
      val fullMessage = s"$message ${sourceInfo.makeMessage()}" +
        (if (extraContext.nonEmpty) s"\n${extraContext.mkString("\n")}" else "")
      new FailedExpectationException(observed, expected, fullMessage)
    }
  }

  implicit class testableClock(clock: Clock) extends AnyTestableData[Clock] {
    val data = clock

    def step(cycles: Int = 1): Unit = {
      simulatedModule.willEvaluate()
      if (cycles == 0) {
        simulatedModule.controller.run(0)
      } else {
        simulationPort.tick(
          timestepsPerPhase = 1,
          maxCycles = cycles,
          inPhaseValue = 0,
          outOfPhaseValue = 1,
          sentinel = None
        )
      }
    }

    /** Ticks this clock up to `maxCycles`.
      *
      * Stops early if the `sentinelPort` is equal to the `sentinelValue`.
      */
    def stepUntil(sentinelPort: Data, sentinelValue: BigInt, maxCycles: Int): Unit = {
      simulatedModule.willEvaluate()
      simulationPort.tick(
        timestepsPerPhase = 1,
        maxCycles = maxCycles,
        inPhaseValue = 0,
        outOfPhaseValue = 1,
        sentinel = Some(simulatedModule.port(sentinelPort), sentinelValue)
      )
    }
  }

  trait AnyTestableData[T <: Data] {
    protected def data: T

    protected val simulatedModule = AnySimulatedModule.current

    protected final def simulationPort = simulatedModule.port(data)
  }

  trait Peekable[T <: Data] extends AnyTestableData[T] {
    def peek(): T

    protected def isSigned = false

    def expect(expected: T, buildMessage: (T, T) => String)(implicit sourceInfo: SourceInfo): Unit

    def expect(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, (_, _) => message)

    def expect(expected: T)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, (observed, expected) => s"Expectation failed: observed value $observed != $expected")

    def expect[U](
      expected:     U,
      encode:       (Simulation.Value) => U,
      buildMessage: (U, U) => String,
      sourceInfo:   SourceInfo
    ): Unit = {
      simulatedModule.willPeek()

      simulationPort.check(isSigned = isSigned) { observedValue =>
        val observed = encode(observedValue)
        if (observed != expected) {
          val extraContext =
            sourceInfo match {
              case sl: SourceLine =>
                ExceptionHelpers.getErrorLineInFile(Seq(), sl)
              case _ =>
                Seq()
            }
          throw FailedExpectationException(
            observed,
            expected,
            buildMessage(observed, expected),
            sourceInfo,
            extraContext
          )
        }
      }
    }
  }

  trait Pokable[T <: Data] extends AnyTestableData[T] {
    def poke(literal: T): Unit
  }

  trait PeekPokable[T <: Data] extends Peekable[T] with Pokable[T]

  sealed trait TestableElement[T <: Element] extends PeekPokable[T] {

    private[simulator] def encode(width: Int, value: BigInt): T

    protected final def encode(value: Simulation.Value): T = {
      encode(value.bitCount, value.asBigInt)
    }

    protected def peekValue(): Simulation.Value = {
      simulatedModule.willPeek()
      simulationPort.get(isSigned = isSigned)
    }

    def peek(): T = encode(peekValue())

    def poke(literal: T): Unit = poke(literal.litValue)

    def poke(value: BigInt): Unit = {
      simulatedModule.willPoke()
      simulationPort.set(value)
    }

    final override def expect(expected: T)(implicit sourceInfo: SourceInfo): Unit = {
      expect(
        expected,
        (observed, expected) => s"Expectation failed: observed value $observed != $expected"
      )
    }

    override def expect(expected: T, buildMessage: (T, T) => String)(implicit sourceInfo: SourceInfo): Unit = {
      require(expected.isLit, s"Expected value: $expected must be a literal")

      simulatedModule.willPeek()

      simulationPort.check(isSigned = isSigned) { observedValue =>
        val observed = encode(observedValue)
        if (observedValue.asBigInt != expected.litValue) {
          val extraContext =
            sourceInfo match {
              case sl: SourceLine =>
                ExceptionHelpers.getErrorLineInFile(Seq(), sl)
              case _ =>
                Seq()
            }
          throw FailedExpectationException(
            observed,
            expected,
            buildMessage(observed, expected),
            sourceInfo,
            extraContext
          )
        }
      }
    }

    final def expect(expected: BigInt)(implicit sourceInfo: SourceInfo): Unit = {
      expect(
        expected,
        _.asBigInt,
        (observed: BigInt, expected: BigInt) => s"Expectation failed: observed value $observed != $expected",
        sourceInfo
      )
    }

    final def expect(expected: BigInt, message: String)(implicit sourceInfo: SourceInfo): Unit = {
      expect(expected, _.asBigInt, (_: BigInt, _: BigInt) => message, sourceInfo)
    }

  }

  implicit final class testableSInt(val data: SInt) extends TestableElement[SInt] {
    override def isSigned = true

    override def encode(width: Int, value: BigInt) = value.asSInt(width.W)
  }

  implicit final class testableUInt(val data: UInt) extends TestableElement[UInt] {
    override def encode(width: Int, value: BigInt) = value.asUInt(width.W)
  }

  implicit final class testableBool(val data: Bool) extends TestableElement[Bool] {
    override def encode(width: Int, value: BigInt): Bool = {
      require(width <= 1, "Bool must have width 1")
      if (value.isValidByte) {
        value.byteValue match {
          case 0 => false.B
          case 1 => true.B
          case x => throw new Exception(s"peeked Bool with value $x, not 0 or 1")
        }
      } else {
        throw new Exception(s"peeked Bool with value $value, not 0 or 1")
      }
    }

    def poke(value: Boolean): Unit = poke(value.B)

    def peekBoolean(): Boolean = peek().litToBoolean

    def expect(value: Boolean)(implicit sourceInfo: SourceInfo): Unit = expect(value.B)
  }

  implicit final class testablReset(val data: Reset) extends TestableElement[Reset] {
    def poke(value: Boolean): Unit = poke(value.B)

    def encode(width: Int, value: BigInt): Reset = testableBool(data.asBool).encode(width, value)
  }

  implicit class testableEnum[T <: EnumType](val data: T) extends TestableElement[T] {
    override def encode(width: Int, value: BigInt): T = {
      data.factory.all.find(_.litValue == value).get.asInstanceOf[T]
    }

    override def expect(expected: T, buildMessage: (T, T) => String)(implicit sourceInfo: SourceInfo): Unit = {
      require(expected.isLit, "Expected value must be a literal")
      val observedSimValue = peekValue()
      val observedVal = observedSimValue.asBigInt
      val expectedVal = expected.litValue
      if (observedVal != expectedVal) {
        throw FailedExpectationException(observedVal, expectedVal, buildMessage(peek(), expected))
      }
    }
  }

  implicit class testableRecord[T <: Record](val data: T)(implicit sourceInfo: SourceInfo) extends PeekPokable[T] {

    override def peek(): T = {
      val elementValueFns = data.elements.map { case (name: String, elt: Data) =>
        (y: Record) =>
          (
            y.elements(name), {
              new testableData(elt).peek()
            }
          )
      }.toSeq
      chiselTypeOf(data).Lit(elementValueFns: _*)
    }

    override def poke(value: T): Unit = {
      data.elements.foreach { case (name, d) =>
        d.poke(value.elements(name))
      }
    }

    override def expect(expected: T, buildMessage: (T, T) => String)(implicit sourceInfo: SourceInfo): Unit = {
      // TODO: not checking for isLit to allow partially specified expected record
      require(DataMirror.checkTypeEquivalence(data, expected), "Type mismatch")

      // FIXME: I can't understand why but _not_ getting the peeked value as a `val` beforehand results in infinite recursion
      val peekedValue = peek()

      data.elements.foreach { case (name, d) =>
        expected.elements(name) match {
          case DontCare => // missing fields are DontCare
          case ve =>
            d.expect(
              ve,
              (obs, exp) => s"${buildMessage(peekedValue, expected)}:\n Expected value of $name to be $exp, got $obs"
            )
        }
      }
    }
  }

  implicit class testableData[T <: Data](val data: T) extends PeekPokable[T] {

    def peek(): T = {
      data match {
        case x: Bool     => new testableBool(x).peek().asInstanceOf[T]
        case x: UInt     => new testableUInt(x).peek().asInstanceOf[T]
        case x: SInt     => new testableSInt(x).peek().asInstanceOf[T]
        case x: EnumType => new testableEnum(x).peek().asInstanceOf[T]
        case x: Record   => new testableRecord(x).peek().asInstanceOf[T]
        case x: Vec[_] =>
          val elementValueFns = x.getElements.map(_.peek())
          Vec.Lit(elementValueFns: _*).asInstanceOf[T]
        case x => throw new Exception(s"don't know how to peek $x")
      }
    }

    override def expect(
      expected:     T,
      buildMessage: (T, T) => String
    )(implicit sourceInfo: SourceInfo): Unit = {

      def buildMsgFn[S](observed: S, expected: S): String =
        buildMessage(observed.asInstanceOf[T], expected.asInstanceOf[T])

      (data, expected) match {
        case (dat: Bool, exp: Bool) =>
          new testableBool(dat).expect(exp, buildMsgFn _)
        case (dat: UInt, exp: UInt) =>
          new testableUInt(dat).expect(exp, buildMsgFn _)
        case (dat: SInt, exp: SInt) =>
          new testableSInt(dat).expect(exp, buildMsgFn _)
        case (dat: EnumType, exp: EnumType) =>
          new testableEnum(dat).expect(exp, buildMsgFn _)
        case (dat: Record, exp: Record) =>
          new testableRecord(dat).expect(exp, buildMsgFn _)
        case (dat: Vec[_], exp: Vec[_]) =>
          require(DataMirror.checkTypeEquivalence(dat, exp), s"Vec type mismatch")
          require( // TODO: this should'nt really be needed, right?
            exp.length == dat.length,
            s"Vec length mismatch: Data port has ${dat.length} elements while the expected value is of length ${exp.length}"
          )
          val peekedValue = dat.peek()
          dat.getElements.zip(exp.getElements).zipWithIndex.foreach { case ((datEl, valEl), index) =>
            valEl match {
              case DontCare =>
              // TODO: missing elements?
              case ve =>
                datEl.expect(
                  ve,
                  (o, e) => buildMessage(peekedValue.asInstanceOf[T], exp.asInstanceOf[T]) + s" at index $index"
                )
            }
          }
        case x => throw new Exception(s"don't know how to expect $x")
      }
    }

    def poke(literal: T): Unit = (data, literal) match {
      case (x: Bool, lit: Bool)         => x.poke(lit)
      case (x: UInt, lit: UInt)         => x.poke(lit)
      case (x: SInt, lit: SInt)         => x.poke(lit)
      case (x: EnumType, lit: EnumType) => x.poke(lit)
      case (x: Record, lit: Record)     => x.poke(lit)
      case (x: Vec[_], lit: Vec[_]) =>
        require(DataMirror.checkTypeEquivalence(x, lit), "Type mismatch")
        require(x.length == lit.length, s"Vec length mismatch: expected ${x.length}, got ${lit.length}")
        x.getElements.zip(lit.getElements).foreach { case (portEl, valueEl) =>
          portEl.poke(valueEl)
        }
      case x => throw new Exception(s"don't know how to poke $x")
    }
  }
}
