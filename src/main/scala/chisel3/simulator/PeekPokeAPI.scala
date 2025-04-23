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

trait Peekable[T <: Data] {

  /**
    * Peek the value of a data port.
    *
    * @return the value of the data port
    */
  def peek(): T

  /**
    * Expect the value of a data port to be equal to the expected value.
    *
    * @param expected the expected value
    * @param buildMessage a function that builds a message for the failure case
    */
  def expect(expected: T, buildMessage: (String, T) => String)(implicit sourceInfo: SourceInfo): Unit

  /**
    * Expect the value of a data port to be equal to the expected value.
    *
    * @param expected the expected value
    * @param message a message for the failure case
    */
  def expect(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit =
    expect(expected, (_, _) => message)

  /**
      * Expect the value of a data port to be equal to the expected value.
      *
      * @param expected the expected value
      */
  def expect(expected: T)(implicit sourceInfo: SourceInfo): Unit =
    expect(expected, (observed, expected) => s"Expectation failed: observed value $observed != $expected")
}

trait Pokable[T <: Data] {

  /**
    * Poke (set) the value of a data port.
    *
    * @param literal the value to set, which must be a literal
    */
  def poke(literal: T): Unit
}

sealed trait AnyTestableData[T <: Data] {
  protected def data: T

  protected def simulatedModule = AnySimulatedModule.current

  protected def simulationPort = simulatedModule.port(data)
}

trait PeekPokable[T <: Data] extends Peekable[T] with Pokable[T] with AnyTestableData[T]

/**
  * Exception thrown when an expectation fails.
  *
  * @param observed the observed value
  * @param expected the expected value
  * @param message the message to display
  */
case class FailedExpectationException[T <: Serializable](observed: T, expected: T, message: String)
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
  def apply[T <: Serializable](
    observed:   T,
    expected:   T,
    message:    String,
    sourceInfo: SourceInfo
  ): FailedExpectationException[T] = {
    val extraContext =
      sourceInfo match {
        case sl: SourceLine =>
          ExceptionHelpers.getErrorLineInFile(Seq(), sl)
        case _ =>
          Seq()
      }
    val fullMessage = s"$message ${sourceInfo.makeMessage()}" +
      (if (extraContext.nonEmpty) s"\n${extraContext.mkString("\n")}" else "")
    new FailedExpectationException[T](observed, expected, fullMessage)
  }
}

object PeekPokeAPI extends PeekPokeAPI

trait PeekPokeAPI {

  implicit class TestableClock(clock: Clock) extends AnyTestableData[Clock] {
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

  sealed trait TestableElement[T <: Element] extends PeekPokable[T] {
    protected def isSigned = false

    private[simulator] protected def encode(width: Int, value: BigInt): T

    private[simulator] final def encode(value: Simulation.Value): T = {
      encode(value.bitCount, value.asBigInt)
    }

    final def peekValue(): Simulation.Value = {
      simulatedModule.willPeek()
      simulationPort.get(isSigned = isSigned)
    }

    def peek(): T = encode(peekValue())

    def poke(literal: T): Unit = poke(literal.litValue)

    def poke(value: BigInt): Unit = {
      simulatedModule.willPoke()
      simulationPort.set(value)
    }

    private[simulator] protected def check[U](checkFn: Simulation.Value => Unit): Unit = {
      simulatedModule.willPeek()
      simulationPort.check(isSigned = isSigned)(checkFn)
    }

    protected final def expect[U](
      expected:       U,
      sameValue:      (Simulation.Value, U) => Boolean,
      formatObserved: (Simulation.Value) => String,
      formatExpected: U => String,
      buildMessage:   (Simulation.Value, U) => String,
      sourceInfo:     SourceInfo
    ): Unit = {
      check(observedValue =>
        if (!sameValue(observedValue, expected)) {
          throw FailedExpectationException(
            formatObserved(observedValue),
            formatExpected(expected),
            buildMessage(observedValue, expected),
            sourceInfo
          )
        }
      )
    }

    protected final def expect[U](
      expected:       U,
      sameValue:      (Simulation.Value, U) => Boolean,
      formatObserved: (Simulation.Value) => String,
      formatExpected: U => String,
      sourceInfo:     SourceInfo
    ): Unit = expect[U](
      expected,
      sameValue,
      formatObserved,
      formatExpected,
      (observedValue: Simulation.Value, expected: U) =>
        s"Expectation failed: observed value ${formatObserved(observedValue)} != ${formatExpected(expected)}",
      sourceInfo
    )

    protected final def expect[U](
      expected:     U,
      sameValue:    (Simulation.Value, U) => Boolean,
      buildMessage: (Simulation.Value, U) => String,
      sourceInfo:   SourceInfo
    ): Unit = expect[U](
      expected,
      sameValue,
      (observedValue: Simulation.Value) => encode(observedValue).toString,
      (expected: U) => expected.toString,
      buildMessage,
      sourceInfo
    )

    override def expect(expected: T)(implicit sourceInfo: SourceInfo): Unit = {
      require(expected.isLit, s"Expected value: $expected must be a literal")
      expect(
        expected,
        (observed: Simulation.Value, expected: T) => observed.asBigInt == expected.litValue,
        formatObserved = (obs: Simulation.Value) => encode(obs).toString,
        formatExpected = (exp: T) => exp.toString,
        sourceInfo = sourceInfo
      )
    }

    override def expect(expected: T, buildMessage: (String, T) => String)(implicit sourceInfo: SourceInfo): Unit = {
      require(expected.isLit, s"Expected value: $expected must be a literal")
      expect[T](
        expected,
        (observed: Simulation.Value, expected: T) => observed.asBigInt == expected.litValue,
        buildMessage = (obs: Simulation.Value, exp: T) => buildMessage(encode(obs).toString, exp),
        sourceInfo = sourceInfo
      )
    }

    final def expect(expected: BigInt, buildMessage: (BigInt, BigInt) => String)(
      implicit sourceInfo: SourceInfo
    ): Unit = expect[BigInt](
      expected,
      (obs: Simulation.Value, exp: BigInt) => obs.asBigInt == exp,
      buildMessage = (obs: Simulation.Value, exp: BigInt) => buildMessage(obs.asBigInt, exp),
      sourceInfo = sourceInfo
    )

    final def expect(expected: BigInt)(implicit sourceInfo: SourceInfo): Unit = expect[BigInt](
      expected,
      (obs: Simulation.Value, exp: BigInt) => obs.asBigInt == exp,
      formatObserved = (obs: Simulation.Value) => obs.asBigInt.toString,
      formatExpected = (exp: BigInt) => exp.toString,
      sourceInfo = sourceInfo
    )

    final def expect(expected: BigInt, message: String)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, (_: BigInt, _: BigInt) => message)

  }

  implicit final class TestableSInt(val data: SInt) extends TestableElement[SInt] {
    override def isSigned = true

    override def encode(width: Int, value: BigInt) = value.asSInt(width.W)
  }

  implicit final class TestableUInt(val data: UInt) extends TestableElement[UInt] {
    override def encode(width: Int, value: BigInt) = value.asUInt(width.W)
  }

  implicit final class TestableBool(val data: Bool) extends TestableElement[Bool] {
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

    override def expect(expected: Bool)(implicit sourceInfo: SourceInfo): Unit = expect[Bool](
      expected,
      (obs: Simulation.Value, exp: Bool) => obs.asBigInt == exp.litValue,
      formatObserved = (obs: Simulation.Value) => obs.asBigInt.toString,
      formatExpected = (exp: Bool) => exp.litValue.toString,
      sourceInfo = sourceInfo
    )

    def expect(value: Boolean)(implicit sourceInfo: SourceInfo): Unit = expect(value.B)
  }

  implicit final class TestableReset(val data: Reset) extends TestableElement[Reset] {
    def poke(value: Boolean): Unit = poke(value.B)

    def encode(width: Int, value: BigInt): Reset = TestableBool(data.asBool).encode(width, value)
  }

  implicit class TestableEnum[T <: EnumType](val data: T) extends TestableElement[T] {
    override def encode(width: Int, value: BigInt): T = {
      data.factory.all.find(_.litValue == value).get.asInstanceOf[T]
    }
  }

  implicit class TestableRecord[T <: Record](val data: T)(implicit sourceInfo: SourceInfo) extends PeekPokable[T] {

    override def peek(): T = {
      chiselTypeOf(data).Lit(
        data.elements.toSeq.map { case (name: String, elt: Data) =>
          (rec: Record) => rec.elements(name) -> elt.peek()
        }: _*
      )
    }

    override def poke(value: T): Unit = data.elements.foreach { case (name, d) =>
      d.poke(value.elements(name))
    }

    def expect(expected: T, buildMessage: (String, T) => String, allowPartial: Boolean)(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      require(DataMirror.checkTypeEquivalence(data, expected), "Type mismatch")

      // FIXME: I can't understand why but _not_ getting the peeked value as a `val` beforehand results in infinite recursion
      val peekedValue = peek().toString

      data.elements.foreach { case (name, d) =>
        expected.elements(name) match {
          case DontCare =>
            if (!allowPartial) {
              throw new Exception(
                s"Field '$name' is not initiazlized in the expected value $expected"
              )
            }
          case exp =>
            d.expect(
              exp,
              (obs: String, _) =>
                s"${buildMessage(peekedValue, expected)}:\n Expected value of field '$name' to be $exp, got $obs"
            )
        }
      }
    }

    override def expect(expected: T, buildMessage: (String, T) => String)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, buildMessage, allowPartial = false)
  }

  implicit class TestableData[T <: Data](val data: T) extends PeekPokable[T] {

    def peek(): T = {
      data match {
        case x: Bool     => new TestableBool(x).peek().asInstanceOf[T]
        case x: UInt     => new TestableUInt(x).peek().asInstanceOf[T]
        case x: SInt     => new TestableSInt(x).peek().asInstanceOf[T]
        case x: EnumType => new TestableEnum(x).peek().asInstanceOf[T]
        case x: Record   => new TestableRecord(x).peek().asInstanceOf[T]
        case x: Vec[_] =>
          val elementValueFns = x.getElements.map(_.peek())
          Vec.Lit(elementValueFns: _*).asInstanceOf[T]
        case x => throw new Exception(s"don't know how to peek $x")
      }
    }

    override def expect(
      expected:     T,
      buildMessage: (String, T) => String
    )(implicit sourceInfo: SourceInfo): Unit = {

      def buildMsgFn[S](observed: String, expected: S): String =
        buildMessage(observed, expected.asInstanceOf[T])

      (data, expected) match {
        case (dat: Bool, exp: Bool) =>
          new TestableBool(dat).expect(exp, buildMsgFn _)
        case (dat: UInt, exp: UInt) =>
          new TestableUInt(dat).expect(exp, buildMsgFn _)
        case (dat: SInt, exp: SInt) =>
          new TestableSInt(dat).expect(exp, buildMsgFn _)
        case (dat: EnumType, exp: EnumType) =>
          new TestableEnum(dat).expect(exp, buildMsgFn _)
        case (dat: Record, exp: Record) =>
          new TestableRecord(dat).expect(exp, buildMsgFn _)
        case (dat: Vec[_], exp: Vec[_]) =>
          require(
            exp.length == dat.length,
            s"Vec length mismatch: Data port has ${dat.length} elements while the expected value is of length ${exp.length}"
          )
          val peekedValue = dat.peek().toString
          dat.getElements.zip(exp.getElements).zipWithIndex.foreach { case ((datEl, valEl), index) =>
            valEl match {
              case DontCare =>
              // TODO: missing elements?
              case ve =>
                datEl.expect(
                  ve,
                  (obs: String, exp: Data) => buildMessage(peekedValue, exp.asInstanceOf[T]) + s" at index $index"
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
        require(x.length == lit.length, s"Vec length mismatch: expected ${x.length}, got ${lit.length}")
        x.getElements.zip(lit.getElements).foreach { case (portEl, valueEl) =>
          portEl.poke(valueEl)
        }
      case x => throw new Exception(s"don't know how to poke $x")
    }
  }
}
