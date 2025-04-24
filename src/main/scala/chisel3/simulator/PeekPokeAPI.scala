package chisel3.simulator

import chisel3._
import chisel3.experimental.BundleLiterals._
import chisel3.experimental.VecLiterals._
import chisel3.experimental.{SourceInfo, SourceLine}
import chisel3.internal.ExceptionHelpers
import chisel3.reflect.DataMirror
import firrtl.options.StageUtils.dramaticMessage
import svsim._

import scala.util.control.NoStackTrace

trait Peekable[T <: Data] {

  /**
    * Get the value of a data port as a literal.
    *
    * @return the value of the peeked data
    */
  def peek(): T

  /**
    * Expect the value of a data port to be equal to the expected value.
    *
    * @param expected the expected value
    * @param buildMessage a function that builds a message for the failure case
    * @throws FailedExpectationException if the observed value does not match the expected value
    */
  def expect(expected: T, buildMessage: (T, T) => String)(implicit sourceInfo: SourceInfo): Unit

  /**
    * Expect the value of a data port to be equal to the expected value.
    *
    * @param expected the expected value
    * @param message a message for the failure case
    * @throws FailedExpectationException if the observed value does not match the expected value
    */
  def expect(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit =
    expect(expected, (_, _) => message)

  /**
      * Expect the value of a data port to be equal to the expected value.
      *
      * @param expected the expected value
      * @throws FailedExpectationException if the observed value does not match the expected value
      */
  def expect(expected: T)(implicit sourceInfo: SourceInfo): Unit =
    expect(expected, (observed, expected) => s"Expectation failed: observed value $observed != $expected")

  private[simulator] final def dataToString(data: Data): String = {
    data match {
      case x: Bundle =>
        x.elements.map { case (name, elt) =>
          s"$name: ${dataToString(elt)}"
        }.mkString("{", ", ", "}")
      case x: Vec[_]   => x.getElements.map(dataToString).mkString("[", ", ", "]")
      case x: EnumType => x.toString
      case x if x.isLit => x.litValue.toString
      case _            => data.toString
    }
  }
}

trait Pokable[T <: Data] {

  /**
    * Sets the value of a data port.
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

sealed trait TestableElement[T <: Element] extends PeekPokable[T] {
  protected def isSigned = false

  private[simulator] def encode(width: Int, value: BigInt): T

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

  private[simulator] def check[U](checkFn: Simulation.Value => Unit): Unit = {
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

  def expect(expected: T, buildMessage: (T, T) => String)(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    require(expected.isLit, s"Expected value: $expected must be a literal")
    expect[T](
      expected,
      (observed: Simulation.Value, expected: T) => observed.asBigInt == expected.litValue,
      buildMessage = (obs: Simulation.Value, exp: T) => buildMessage(encode(obs), exp),
      formatObserved = (obs: Simulation.Value) => encode(obs).toString,
      formatExpected = (exp: T) => exp.toString,
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

object PeekPokeAPI {

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

  implicit class TestableRecord[T <: Record](val data: T) extends PeekPokable[T] {

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

    def expect(expected: T, buildMessage: (T, T, String) => String, allowPartial: Boolean)(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      require(DataMirror.checkTypeEquivalence(data, expected), "Type mismatch")

      // FIXME: I can't understand why but _not_ getting the peeked value as a `val` beforehand results in infinite recursion
      val peekedValue = peek()

      data.elements.foreach { case (elName, elData) =>
        expected.elements(elName) match {
          case DontCare =>
            if (!allowPartial) {
              throw new Exception(
                s"Field '$elName' is not initialized in the expected value $expected"
              )
            }
          case elExp =>
            elData.expect(
              elExp,
              (_, _) => buildMessage(peekedValue, expected, elName)
            )
        }
      }
    }

    def expect(expected: T, buildMessage: (T, T) => String, allowPartial: Boolean)(
      implicit sourceInfo: SourceInfo
    ): Unit = expect(
      expected,
      (observed: T, _: T, elName: String) => {
        val userMsg = buildMessage(observed, expected)
        val msg = if (userMsg.nonEmpty) s"${userMsg}:\n" else ""
        val expEl = expected.elements(elName)
        val obsEl = observed.elements(elName)
        s"${msg}Expected the value of element '$elName' to be ${dataToString(expEl)}, got ${dataToString(obsEl)}"
      },
      allowPartial = allowPartial
    )

    def expect(expected: T, buildMessage: (T, T) => String)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, (observed: T, _: T) => buildMessage(observed, expected), allowPartial = false)

    def expect(expected: T, allowPartial: Boolean)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, (obs: T, exp: T) => "", allowPartial)

    override def expect(expected: T)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, (obs: T, exp: T) => "", allowPartial = false)
  }

  implicit class TestableVec[T <: Data](val data: Vec[T]) extends PeekPokable[Vec[T]] {
    override def peek(): Vec[T] = {
      val elementValues = data.getElements.map(_.peek().asInstanceOf[T])
      chiselTypeOf(data).Lit(elementValues.zipWithIndex.map { _.swap }: _*)
    }

    override def poke(literal: Vec[T]): Unit = {
      require(data.length == literal.length, s"Vec length mismatch: expected ${data.length}, got ${literal.length}")
      data.getElements.zip(literal.getElements).foreach {
        case (portEl, valueEl) if portEl.getClass == valueEl.getClass =>
          portEl.poke(valueEl)
        case (portEl, valueEl) =>
          throw new Exception(
            s"This should never happen! Port element type: ${portEl.getClass} != literal element ${valueEl.getClass}"
          )
      }
    }

    final def expectVec[U <: Data](expected: Vec[U], buildMessage: (Vec[T], Vec[U], Int) => String)(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      require(
        expected.length == data.length,
        s"Vec length mismatch: Data port has ${data.length} elements while the expected value is of length ${expected.length}"
      )
      val peekedValue = data.peek()
      data.zip(expected).zipWithIndex.foreach { case ((datEl, expEl), index) =>
        expEl match {
          case DontCare =>
          // TODO: missing elements?
          case exp if exp.getClass == datEl.getClass =>
            datEl.expect(
              exp.asInstanceOf[T],
              (obs: T, exp: T) => buildMessage(peekedValue, expected, index)
            )
          case _ =>
            throw new Exception(
              s"Type mismatch: ${expEl.getClass} != ${datEl.getClass}"
            )
        }
      }
    }

    override def expect(expected: Vec[T], buildMessage: (Vec[T], Vec[T]) => String)(
      implicit sourceInfo: SourceInfo
    ): Unit =
      expectVec[T](
        expected,
        (observed: Vec[T], expected: Vec[T], idx: Int) =>
          buildMessage(observed, expected) + s". Mismatch at index $idx."
      )
  }

  implicit class TestableData[T <: Data](val data: T) extends PeekPokable[T] {

    def peek(): T = {
      data match {
        case x: Bool     => new TestableBool(x).peek().asInstanceOf[T]
        case x: UInt     => new TestableUInt(x).peek().asInstanceOf[T]
        case x: SInt     => new TestableSInt(x).peek().asInstanceOf[T]
        case x: EnumType => new TestableEnum(x).peek().asInstanceOf[T]
        case x: Record   => new TestableRecord(x).peek().asInstanceOf[T]
        case x: Vec[_]   => new TestableVec(x).peek().asInstanceOf[T]
        case x => throw new Exception(s"don't know how to peek $x")
      }
    }

    override def expect(
      expected:     T,
      buildMessage: (T, T) => String
    )(implicit sourceInfo: SourceInfo): Unit = {

      def buildMsgFn[S](obs: S, exp: S): String = {
        require(obs.getClass == exp.getClass, s"Type mismatch: ${obs.getClass} != ${exp.getClass}")
        buildMessage(obs.asInstanceOf[T], exp.asInstanceOf[T])
      }

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
          new TestableVec(dat).expectVec(
            exp,
            (obs: Data, _: Data, idx: Int) => {
              require(obs.getClass == exp.getClass, s"Type mismatch: ${obs.getClass} != ${exp.getClass}")
              buildMessage(obs.asInstanceOf[T], expected.asInstanceOf[T]) + s". Mismatch at index $idx."
            }
          )
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
          require(
            portEl.getClass == valueEl.getClass,
            s"Vec element type mismatch: expected ${portEl.getClass}, got ${valueEl.getClass}"
          )
          portEl.poke(valueEl)
        }
      case x => throw new Exception(s"Don't know how to poke $x")
    }
  }
}

trait PeekPokeAPI {
  import scala.language.implicitConversions

  implicit def toTestableClock(clock: Clock): PeekPokeAPI.TestableClock = new PeekPokeAPI.TestableClock(clock)

  implicit def toTestableSInt(sint: SInt): PeekPokeAPI.TestableSInt = new PeekPokeAPI.TestableSInt(sint)

  implicit def toTestableUInt(uint: UInt): PeekPokeAPI.TestableUInt = new PeekPokeAPI.TestableUInt(uint)

  implicit def toTestableBool(bool: Bool): PeekPokeAPI.TestableBool = new PeekPokeAPI.TestableBool(bool)

  implicit def toTestableReset(reset: Reset): PeekPokeAPI.TestableReset = new PeekPokeAPI.TestableReset(reset)

  implicit def toTestableEnum[T <: EnumType](data: T): PeekPokeAPI.TestableEnum[T] = new PeekPokeAPI.TestableEnum(data)

  implicit def toTestableRecord[T <: Record](record: T): PeekPokeAPI.TestableRecord[T] =
    new PeekPokeAPI.TestableRecord(record)

  implicit def toTestableVec[T <: Data](vec: Vec[T]): PeekPokeAPI.TestableVec[T] = new PeekPokeAPI.TestableVec(vec)

  implicit def toTestableData[T <: Data](data: T): PeekPokeAPI.TestableData[T] = new PeekPokeAPI.TestableData(data)
}
