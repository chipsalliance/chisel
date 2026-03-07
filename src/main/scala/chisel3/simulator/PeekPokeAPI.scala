package chisel3.simulator

import chisel3._
import chisel3.experimental.BundleLiterals
import chisel3.experimental.VecLiterals._
import chisel3.experimental.{SourceInfo, SourceLine}
import chisel3.internal.ExceptionHelpers
import chisel3.internal.binding.DontCareBinding
import firrtl.options.StageUtils.dramaticMessage
import svsim._

import scala.util.control.NoStackTrace
import scala.language.implicitConversions

trait Peekable[T <: Data] {

  /**
    * Get the value of a data port as a literal.
    *
    * @return the value of the peeked data
    */
  def peek()(implicit sourceInfo: SourceInfo): T

  /**
      * Expect the value of a data port to be equal to the expected value.
      *
      * @param expected the expected value
      * @throws FailedExpectationException if the observed value does not match the expected value
      */
  def expect(expected: T)(implicit sourceInfo: SourceInfo): Unit = expect(expected, "")

  /**
  * Expect the value of a data port to be equal to the expected value.
  *
  * @param expected the expected value
  * @param message a message for the failure case
  * @throws FailedExpectationException if the observed value does not match the expected value
  */
  def expect(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit

  /**
  * Expect the value of a data port to be equal to the expected value.
  *
  * @param expected the expected value
  * @param format formatting strategy for rendered values in failure output
  * @throws FailedExpectationException if the observed value does not match the expected value
  */
  def expect(expected: T, format: ExpectationValueFormat.Type)(implicit sourceInfo: SourceInfo): Unit =
    expect(expected, "", format)

  /**
  * Expect the value of a data port to be equal to the expected value.
  *
  * @param expected the expected value
  * @param message a message for the failure case
  * @param format formatting strategy for rendered values in failure output
  * @throws FailedExpectationException if the observed value does not match the expected value
  */
  def expect(expected: T, message: String, format: ExpectationValueFormat.Type)(implicit sourceInfo: SourceInfo): Unit =
    expect(expected, message)

  private[simulator] def dataToString(value: Data): String = {
    value match {
      case x: Bundle =>
        x.elements.map { case (name, elt) =>
          s"$name: ${dataToString(elt)}"
        }.mkString("{", ", ", "}")
      case x: Vec[_]   => x.getElements.map(dataToString).mkString("[", ", ", "]")
      case x: EnumType => x.toString
      case x if x.isLit => x.litValue.toString
      case _            => value.toString
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

  protected def simulatedModule: AnySimulatedModule = AnySimulatedModule.current

  protected def simulationPort: Simulation.Port = simulatedModule.port(data)
}

trait PeekPokable[T <: Data] extends Peekable[T] with Pokable[T] with AnyTestableData[T]

trait PeekPokeApiException extends NoStackTrace

/** Formatting options for values rendered by `expect` failure messages.
  */
object ExpectationValueFormat {

  /** Context passed to custom formatters.
    *
    * @param chiselType Chisel data type string (e.g. `UInt<32>`)
    * @param signedValue Numeric value as observed by the simulator for this data type
    * @param bitWidth Bit width of the rendered value
    * @param isSigned Whether the data type is signed
    * @param displayString Value rendered using the decimal display for this context
    */
  final case class Value(
    chiselType:    String,
    signedValue:   BigInt,
    bitWidth:      Int,
    isSigned:      Boolean,
    displayString: String
  ) {

    /** Value interpreted as an unsigned integer with `bitWidth` bits. */
    val unsignedValue: BigInt = {
      if (bitWidth <= 0) {
        signedValue
      } else {
        val mask = (BigInt(1) << bitWidth) - 1
        signedValue & mask
      }
    }
  }

  /** Raw failure context passed to formatters. */
  final case class Failure(observed: Value, expected: Value)

  /** Fully rendered failure output. */
  final case class Rendered(observed: String, expected: String, message: String)

  /** Value format type. */
  sealed trait Type {
    private[simulator] def render(failure: Failure): Rendered
  }

  /** Render values using their decimal display. */
  object Dec extends Type {
    override private[simulator] def render(failure: Failure): Rendered =
      renderValues(failure)(_.displayString)
  }

  /** Render the numeric part as hexadecimal (prefixed with `0x`) and grouped by bytes. */
  object Hex extends Type {
    override private[simulator] def render(failure: Failure): Rendered =
      renderValues(failure)(value => withType(value.chiselType, s"0x${formatGroupedHexDigits(value)}"))
  }

  /** Render the numeric part as binary (prefixed with `0b`). */
  object Bin extends Type {
    override private[simulator] def render(failure: Failure): Rendered =
      renderValues(failure)(value => withType(value.chiselType, s"0b${formatGroupedBinaryDigits(value)}"))
  }

  object Custom {

    /** Use a user-provided formatter for each rendered value. */
    def apply(formatValue: Value => String): Type = values(formatValue)

    /** Use a user-provided formatter for the failure message. */
    def apply(buildMessage: (Value, Value) => String): Type = message(buildMessage)

    /** Use a user-provided formatter for each rendered value. */
    def values(formatValue: Value => String): Type =
      customType { failure =>
        val observed = formatValue(failure.observed)
        val expected = formatValue(failure.expected)
        Rendered(observed, expected, defaultMessage(observed, expected))
      }

    /** Use a user-provided formatter for the failure message while keeping the rendered values. */
    def message(buildMessage: (Value, Value) => String): Type = message(Dec)(buildMessage)

    /** Use a user-provided formatter for the failure message on top of a base value format. */
    def message(base: Type)(buildMessage: (Value, Value) => String): Type =
      customType { failure =>
        val rendered = base.render(failure)
        rendered.copy(message = buildMessage(failure.observed, failure.expected))
      }

    private def customType(renderFailure: Failure => Rendered): Type = new CustomType(renderFailure)
  }

  private final class CustomType(renderFailure: Failure => Rendered) extends Type {
    override private[simulator] def render(failure: Failure): Rendered = renderFailure(failure)
  }

  private def renderValues(failure: Failure)(formatValue: Value => String): Rendered = {
    val observed = formatValue(failure.observed)
    val expected = formatValue(failure.expected)
    Rendered(observed, expected, defaultMessage(observed, expected))
  }

  private def defaultMessage(observed: String, expected: String): String =
    s"Expectation failed: observed value $observed != $expected"

  private def formatGroupedHexDigits(value: Value): String = {
    val minimumDigits = if (value.bitWidth > 0) (value.bitWidth + 3) / 4 else 0
    groupDigits(padLeft(value.unsignedValue.toString(16), minimumDigits), 2)
  }

  private def formatGroupedBinaryDigits(value: Value): String = {
    val minimumDigits = if (value.bitWidth > 0) value.bitWidth else 0
    groupDigits(padLeft(value.unsignedValue.toString(2), minimumDigits), 4)
  }

  private def padLeft(digits: String, minimumWidth: Int): String = {
    if (minimumWidth > digits.length) {
      ("0" * (minimumWidth - digits.length)) + digits
    } else {
      digits
    }
  }

  private def groupDigits(digits: String, groupSize: Int): String =
    digits.reverse.grouped(groupSize).map(_.reverse).toSeq.reverse.mkString(" ")

  private def withType(chiselType: String, value: String): String = {
    if (chiselType.nonEmpty) s"$chiselType($value)" else value
  }
}

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
    with PeekPokeApiException

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

case class UninitializedElementException(message: String)(implicit sourceInfo: SourceInfo)
    extends RuntimeException(
      {
        val extraContext =
          sourceInfo match {
            case sl: SourceLine =>
              ExceptionHelpers.getErrorLineInFile(Seq(), sl)
            case _ =>
              Seq()
          }
        dramaticMessage(
          header = Some("Uninitialized Element"),
          body = s"$message ${sourceInfo.makeMessage()}" +
            (if (extraContext.nonEmpty) s"\n${extraContext.mkString("\n")}" else "")
        )
      }
    )
    with PeekPokeApiException

sealed trait TestableAggregate[T <: Aggregate] extends PeekPokable[T] {

  /**
   * Expect the value of a data port to be equal to the expected value, skipping all uninitialized elements.
   *
   * @param expected the expected value
   *  @param message a message for the failure case
   * @throws FailedExpectationException if the observed value does not match the expected value
   */
  def expectPartial(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit

  def expectPartial(
    expected: T,
    message:  String,
    format:   ExpectationValueFormat.Type
  )(implicit sourceInfo: SourceInfo): Unit =
    expectPartial(expected, message)

  def expectPartial(expected: T, format: ExpectationValueFormat.Type)(implicit sourceInfo: SourceInfo): Unit =
    expectPartial(expected, "", format)

  /**
   * Expect the value of a data port to be equal to the expected value, skipping all uninitialized elements.
   *
   * @param expected the expected value
   * @throws FailedExpectationException if the observed value does not match the expected value
   */
  def expectPartial(expected: T)(implicit sourceInfo: SourceInfo): Unit =
    expectPartial(expected, "", ExpectationValueFormat.Dec)
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

  def peek()(implicit sourceInfo: SourceInfo): T = encode(peekValue())

  def poke(literal: T): Unit = poke(literal.litValue)

  def poke(value: BigInt): Unit = {
    simulatedModule.willPoke()
    simulationPort.set(value)
  }

  private[simulator] def check[U](checkFn: Simulation.Value => Unit): Unit = {
    simulatedModule.willPeek()
    simulationPort.check(isSigned = isSigned)(checkFn)
  }

  private def portBitWidth: Int = if (data.widthKnown) data.getWidth else 0

  private def dataTypeString: String = chiselTypeOf(data).toString

  private def valueContext(
    chiselType:    String,
    signedValue:   BigInt,
    bitWidth:      Int,
    displayString: String
  ): ExpectationValueFormat.Value =
    ExpectationValueFormat.Value(
      chiselType = chiselType,
      signedValue = signedValue,
      bitWidth = bitWidth,
      isSigned = isSigned,
      displayString = displayString
    )

  protected final def valueContextForLiteral(
    literal:     Data,
    signedValue: BigInt,
    bitWidth:    Int
  ): ExpectationValueFormat.Value =
    valueContext(chiselTypeOf(literal).toString, signedValue, bitWidth, literal.toString)

  protected final def valueContextForObservedLiteral(observed: Simulation.Value): ExpectationValueFormat.Value = {
    val encoded = encode(observed)
    valueContextForLiteral(encoded, observed.asBigInt, observed.bitCount)
  }

  protected final def valueContextForExpectedLiteral(
    expected:         T,
    observedBitWidth: Int
  ): ExpectationValueFormat.Value = {
    val effectiveBitWidth = if (portBitWidth > 0) portBitWidth else observedBitWidth
    valueContextForRawValue(expected.litValue, effectiveBitWidth)
  }

  private def displayStringForRawValue(value: BigInt, bitWidth: Int): String = {
    try {
      encode(bitWidth, value).toString
    } catch {
      case _: Exception =>
        if (dataTypeString.nonEmpty) s"$dataTypeString(${value.toString})" else value.toString
    }
  }

  protected final def valueContextForRawValue(value: BigInt, bitWidth: Int): ExpectationValueFormat.Value =
    valueContext(dataTypeString, value, bitWidth, displayStringForRawValue(value, bitWidth))

  protected final def valueContextForRawValue(
    value:         BigInt,
    bitWidth:      Int,
    displayString: String
  ): ExpectationValueFormat.Value =
    valueContext(dataTypeString, value, bitWidth, displayString)

  protected final def valueContextForObservedRaw(observed: Simulation.Value): ExpectationValueFormat.Value =
    valueContextForRawValue(observed.asBigInt, observed.bitCount)

  protected final def valueContextForExpectedRaw(
    expected:         BigInt,
    observedBitWidth: Int
  ): ExpectationValueFormat.Value = {
    val effectiveBitWidth = if (portBitWidth > 0) portBitWidth else observedBitWidth
    valueContextForRawValue(expected, effectiveBitWidth)
  }

  protected final def renderFailureWithFormat(
    observed: ExpectationValueFormat.Value,
    expected: ExpectationValueFormat.Value,
    format:   ExpectationValueFormat.Type
  ): ExpectationValueFormat.Rendered =
    format.render(ExpectationValueFormat.Failure(observed, expected))

  protected final def rawFailureMessage(observed: BigInt, expected: BigInt): String =
    s"Expectation failed: observed value ${observed.toString} != ${expected.toString}"

  protected final def expect[U](
    expected:      U,
    sameValue:     (Simulation.Value, U) => Boolean,
    renderFailure: (Simulation.Value, U) => ExpectationValueFormat.Rendered,
    buildMessage:  (Simulation.Value, U) => String,
    sourceInfo:    SourceInfo
  ): Unit = {
    check(observedValue =>
      if (!sameValue(observedValue, expected)) {
        val rendered = renderFailure(observedValue, expected)
        throw FailedExpectationException(
          rendered.observed,
          rendered.expected,
          buildMessage(observedValue, expected),
          sourceInfo
        )
      }
    )
  }

  protected final def expect[U](
    expected:      U,
    sameValue:     (Simulation.Value, U) => Boolean,
    renderFailure: (Simulation.Value, U) => ExpectationValueFormat.Rendered,
    sourceInfo:    SourceInfo
  ): Unit = {
    check(observedValue =>
      if (!sameValue(observedValue, expected)) {
        val rendered = renderFailure(observedValue, expected)
        throw FailedExpectationException(
          rendered.observed,
          rendered.expected,
          rendered.message,
          sourceInfo
        )
      }
    )
  }

  protected final def expect[U](
    expected:       U,
    sameValue:      (Simulation.Value, U) => Boolean,
    formatObserved: Simulation.Value => String,
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
    formatObserved: Simulation.Value => String,
    formatExpected: U => String,
    sourceInfo:     SourceInfo
  ): Unit = {
    check(observedValue =>
      if (!sameValue(observedValue, expected)) {
        val observed = formatObserved(observedValue)
        val expectedValue = formatExpected(expected)
        throw FailedExpectationException(
          observed,
          expectedValue,
          s"Expectation failed: observed value $observed != $expectedValue",
          sourceInfo
        )
      }
    )
  }

  /**
  * Expect the value of a data port to be equal to the expected value.
  *
  * @param expected the expected value
  * @param buildMessage a function taking (observedValue: T, expectedValue: T) and returning a String message for the failure case
  * @throws FailedExpectationException if the observed value does not match the expected value
  */
  def expect(expected: T, buildMessage: (T, T) => String)(
    implicit sourceInfo: SourceInfo
  ): Unit =
    expect(expected, buildMessage, ExpectationValueFormat.Dec)

  def expect(
    expected:     T,
    buildMessage: (T, T) => String,
    format:       ExpectationValueFormat.Type
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    require(expected.isLit, s"Expected value: $expected must be a literal")
    expect[T](
      expected,
      (observed: Simulation.Value, expected: T) => observed.asBigInt == expected.litValue,
      renderFailure = (obs: Simulation.Value, exp: T) =>
        renderFailureWithFormat(
          valueContextForObservedLiteral(obs),
          valueContextForExpectedLiteral(exp, obs.bitCount),
          format
        ),
      buildMessage = (obs: Simulation.Value, exp: T) => buildMessage(encode(obs), exp),
      sourceInfo = sourceInfo
    )
  }

  override def expect(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit =
    expect(expected, (_: T, _: T) => message, ExpectationValueFormat.Dec)

  override def expect(
    expected: T,
    message:  String,
    format:   ExpectationValueFormat.Type
  )(implicit sourceInfo: SourceInfo): Unit =
    expect(expected, (_: T, _: T) => message, format)

  final def expect(expected: BigInt)(implicit sourceInfo: SourceInfo): Unit =
    expect[BigInt](
      expected,
      (obs: Simulation.Value, exp: BigInt) => obs.asBigInt == exp,
      renderFailure = (obs: Simulation.Value, exp: BigInt) => {
        val rendered = renderFailureWithFormat(
          valueContextForObservedRaw(obs),
          valueContextForExpectedRaw(exp, obs.bitCount),
          ExpectationValueFormat.Dec
        )
        rendered.copy(message = rawFailureMessage(obs.asBigInt, exp))
      },
      sourceInfo = sourceInfo
    )

  final def expect(expected: BigInt, format: ExpectationValueFormat.Type)(implicit sourceInfo: SourceInfo): Unit =
    expect[BigInt](
      expected,
      (obs: Simulation.Value, exp: BigInt) => obs.asBigInt == exp,
      renderFailure = (obs: Simulation.Value, exp: BigInt) =>
        renderFailureWithFormat(valueContextForObservedRaw(obs), valueContextForExpectedRaw(exp, obs.bitCount), format),
      sourceInfo = sourceInfo
    )

  final def expect(expected: BigInt, message: String)(implicit sourceInfo: SourceInfo): Unit =
    expect(expected, message, ExpectationValueFormat.Dec)

  final def expect(
    expected: BigInt,
    message:  String,
    format:   ExpectationValueFormat.Type
  )(implicit sourceInfo: SourceInfo): Unit =
    expect[BigInt](
      expected,
      (obs: Simulation.Value, exp: BigInt) => obs.asBigInt == exp,
      renderFailure = (obs: Simulation.Value, exp: BigInt) =>
        renderFailureWithFormat(valueContextForObservedRaw(obs), valueContextForExpectedRaw(exp, obs.bitCount), format),
      buildMessage = (_: Simulation.Value, _: BigInt) => message,
      sourceInfo = sourceInfo
    )

  override def expect(expected: T, format: ExpectationValueFormat.Type)(implicit sourceInfo: SourceInfo): Unit = {
    require(expected.isLit, s"Expected value: $expected must be a literal")
    expect(
      expected,
      (observed: Simulation.Value, expected: T) => observed.asBigInt == expected.litValue,
      renderFailure = (obs: Simulation.Value, exp: T) =>
        renderFailureWithFormat(
          valueContextForObservedLiteral(obs),
          valueContextForExpectedLiteral(exp, obs.bitCount),
          format
        ),
      sourceInfo = sourceInfo
    )
  }

  override def expect(expected: T)(implicit sourceInfo: SourceInfo): Unit = expect(expected, ExpectationValueFormat.Dec)
}

object PeekPokeAPI {
  implicit class TestableClock(clock: Clock) extends AnyTestableData[Clock] {
    val data = clock

    def step(cycles: Int = 1, period: Int = 10): Unit = {
      require(
        period >= 2,
        s"specified period, '${period}', must be 2 or greater because an integer half period must be non-zero"
      )

      simulatedModule.willEvaluate()
      if (cycles == 0) {
        simulatedModule.controller.run(0)
      } else {
        simulationPort.tick(
          timestepsPerPhase = period / 2,
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
    def stepUntil(sentinelPort: Data, sentinelValue: BigInt, maxCycles: Int, period: Int = 10): Unit = {
      require(
        period >= 2,
        s"specified period, '${period}', must be 2 or greater because an integer half period must be non-zero"
      )

      simulatedModule.willEvaluate()
      simulationPort.tick(
        timestepsPerPhase = period / 2,
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
          case x => throw new Exception(s"encode Bool with value $x, not 0 or 1")
        }
      } else {
        throw new Exception(s"encode Bool with value $value, not 0 or 1")
      }
    }

    def peekBoolean(): Boolean = peekValue().asBigInt == 1

    override def expect(expected: Bool)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, ExpectationValueFormat.Dec)

    override def expect(
      expected: Bool,
      format:   ExpectationValueFormat.Type
    )(implicit sourceInfo: SourceInfo): Unit = {
      require(expected.isLit, s"Expected value: $expected must be a literal")
      expect(
        expected,
        (observed: Simulation.Value, expected: Bool) => observed.asBigInt == expected.litValue,
        renderFailure = (observed: Simulation.Value, expected: Bool) =>
          renderFailureWithFormat(
            valueContextForRawValue(observed.asBigInt, observed.bitCount, observed.asBigInt.toString),
            valueContextForRawValue(expected.litValue, expected.getWidth, expected.litValue.toString),
            format
          ),
        sourceInfo = sourceInfo
      )
    }

    def expect(value: Boolean)(implicit sourceInfo: SourceInfo): Unit = expect(value.B)

    def expect(value: Boolean, format: ExpectationValueFormat.Type)(implicit sourceInfo: SourceInfo): Unit =
      expect(value.B, format)

    def poke(value: Boolean): Unit = poke(if (value) 1 else 0)
  }

  implicit final class TestableReset(val data: Reset) extends TestableElement[Reset] {
    def encode(width: Int, value: BigInt): Reset = TestableBool(data.asBool).encode(width, value)

    def poke(value: Boolean): Unit = poke(if (value) 1 else 0)
  }

  implicit class TestableEnum[T <: EnumType](val data: T) extends TestableElement[T] {
    override def encode(width: Int, value: BigInt): T = {
      data.factory.all.find(_.litValue == value).get.asInstanceOf[T]
    }
  }

  implicit class TestableRecord[T <: Record](val data: T) extends TestableAggregate[T] {
    override def peek()(implicit sourceInfo: SourceInfo): T = {
      chiselTypeOf(data)._makeLit(
        data.elements.toSeq.map { case (name: String, elt: Data) =>
          (rec: T) => rec.elements(name) -> elt.peek()
        }: _*
      )
    }

    def expect(
      expected:     T,
      buildMessage: (T, T, String) => String,
      allowPartial: Boolean = false
    )(
      implicit sourceInfo: SourceInfo
    ): Unit =
      expect(expected, buildMessage, ExpectationValueFormat.Dec, allowPartial)

    def expect(
      expected:     T,
      buildMessage: (T, T, String) => String,
      format:       ExpectationValueFormat.Type,
      allowPartial: Boolean
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      data.elements.foreach { case (elName, portEl) =>
        expected.elements(elName) match {
          case expEl: Element if expEl.topBindingOpt == Some(DontCareBinding()) =>
            if (!allowPartial) {
              throw new UninitializedElementException(
                s"Element '$elName' in the expected value is not initialized"
              )
            }
          case expEl if expEl.getClass == portEl.getClass =>
            // Not peeking the value beforehand or using `def` or `lazy val` results in a mysterious infinite recursion and StackOverflowError
            // The value is not used if the expected value matches and incurs an overhead
            // TODO: dig deeper amd see if we can avoid this
            val message = buildMessage(peek(), expected, elName)
            (allowPartial, portEl) match {
              case (true, rec: Record) =>
                rec.expectPartial(expEl.asInstanceOf[rec.type], message, format)
              case (true, vec: Vec[_]) =>
                vec.expectPartial(expEl.asInstanceOf[vec.type], message, format)
              case _ =>
                portEl.expect(expEl, message, format)
            }
          case expEl =>
            throw new Exception(
              s"Type mismatch: type of the expected element $elName (${expEl.getClass}) is different from the tested element (${portEl.getClass})"
            )
        }
      }
    }

    private def defaultMessageBuilder(
      observed:    T,
      expected:    T,
      elName:      String,
      userMessage: String = ""
    ): String = (if (userMessage.nonEmpty) s"$userMessage\n" else "") +
      s"Expectation failed for element '$elName': observed value ${dataToString(observed.elements(elName))} != expected value ${dataToString(expected.elements(elName))}"

    override def expectPartial(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, defaultMessageBuilder(_, _, _, message), allowPartial = true)

    override def expectPartial(
      expected: T,
      message:  String,
      format:   ExpectationValueFormat.Type
    )(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, defaultMessageBuilder(_, _, _, message), format = format, allowPartial = true)

    override def expect(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, defaultMessageBuilder(_, _, _, message), allowPartial = false)

    override def expect(
      expected: T,
      message:  String,
      format:   ExpectationValueFormat.Type
    )(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, defaultMessageBuilder(_, _, _, message), format = format, allowPartial = false)

    override def poke(literal: T): Unit = data.elements.foreach { case (name, portEl) =>
      val valueEl = literal.elements(name)
      require(
        portEl.getClass == valueEl.getClass,
        s"Type mismatch for Record element '$name': expected ${portEl.getClass}, got ${valueEl.getClass}"
      )
      portEl.poke(valueEl)
    }
  }

  implicit class TestableVec[T <: Data](val data: Vec[T]) extends TestableAggregate[Vec[T]] {
    override def peek()(implicit sourceInfo: SourceInfo): Vec[T] = {
      val elementValues = data.getElements.map(_.peek().asInstanceOf[T])
      chiselTypeOf(data).Lit(elementValues.zipWithIndex.map { _.swap }: _*)
    }

    override def poke(literal: Vec[T]): Unit = {
      require(data.length == literal.length, s"Vec length mismatch: expected ${data.length}, got ${literal.length}")
      data.getElements.zip(literal).foreach {
        case (portEl, valueEl) if portEl.getClass == valueEl.getClass =>
          portEl.poke(valueEl)
        case (portEl, valueEl) =>
          throw new Exception(
            s"Port element type: ${portEl.getClass} != literal element ${valueEl.getClass}"
          )
      }
    }

    private def defaultMessageBuilder(
      observed:    Vec[T],
      expected:    Vec[T],
      elIndex:     Int,
      userMessage: String = ""
    ): String = (if (userMessage.nonEmpty) s"$userMessage\n" else "") +
      s"Expectation failed for Vec element at index $elIndex: observed value ${dataToString(observed(elIndex))} != expected value ${dataToString(expected(elIndex))}"

    override def expectPartial(expected: Vec[T], message: String)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, defaultMessageBuilder(_, _, _, message), allowPartial = true)

    override def expectPartial(
      expected: Vec[T],
      message:  String,
      format:   ExpectationValueFormat.Type
    )(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, defaultMessageBuilder(_, _, _, message), format = format, allowPartial = true)

    override def expect(expected: Vec[T], message: String)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, defaultMessageBuilder(_, _, _, message), allowPartial = false)

    override def expect(
      expected: Vec[T],
      message:  String,
      format:   ExpectationValueFormat.Type
    )(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, defaultMessageBuilder(_, _, _, message), format = format, allowPartial = false)

    def expect(
      expected:     Vec[T],
      buildMessage: (Vec[T], Vec[T], Int) => String,
      allowPartial: Boolean = false
    )(
      implicit sourceInfo: SourceInfo
    ): Unit =
      expect(expected, buildMessage, ExpectationValueFormat.Dec, allowPartial)

    def expect(
      expected:     Vec[T],
      buildMessage: (Vec[T], Vec[T], Int) => String,
      format:       ExpectationValueFormat.Type,
      allowPartial: Boolean
    )(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      data.getElements.zip(expected).zipWithIndex.foreach {
        case ((datEl: Element, expEl: Element), idx) if expEl.topBindingOpt == Some(DontCareBinding()) =>
          if (!allowPartial)
            throw new UninitializedElementException(
              s"Vec element at index $idx in the expected value is not initialized"
            )
        case ((datEl, expEl), idx) if datEl.getClass == expEl.getClass =>
          val message = buildMessage(peek(), expected, idx)
          (allowPartial, datEl) match {
            case (true, rec: Record) =>
              rec.expectPartial(expEl.asInstanceOf[rec.type], message, format)
            case (true, vec: Vec[_]) =>
              vec.expectPartial(expEl.asInstanceOf[vec.type], message, format)
            case _ =>
              datEl.expect(expEl, message, format)
          }
        case ((datEl, expEl), _) =>
          throw new Exception(
            s"Type mismatch: ${expEl.getClass} != ${datEl.getClass}"
          )
      }
    }
  }

  implicit class TestableData[T <: Data](val data: T) extends PeekPokable[T] {

    private def toPeekable: Peekable[_] = {
      data match {
        case x: Bool     => new TestableBool(x)
        case x: UInt     => new TestableUInt(x)
        case x: SInt     => new TestableSInt(x)
        case x: EnumType => new TestableEnum(x)
        case x: Record   => new TestableRecord(x)
        case x: Vec[_]   => new TestableVec(x)
        case x => throw new Exception(s"don't know how to peek $x")
      }
    }

    def peek()(implicit sourceInfo: SourceInfo): T = toPeekable.peek().asInstanceOf[T]

    override def expect(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit =
      expect(expected, message, ExpectationValueFormat.Dec)

    override def expect(
      expected: T,
      message:  String,
      format:   ExpectationValueFormat.Type
    )(implicit sourceInfo: SourceInfo): Unit = {
      (data, expected) match {
        case (dat: Bool, exp: Bool) =>
          new TestableBool(dat).expect(exp, message, format)
        case (dat: UInt, exp: UInt) =>
          new TestableUInt(dat).expect(exp, message, format)
        case (dat: SInt, exp: SInt) =>
          new TestableSInt(dat).expect(exp, message, format)
        case (dat: EnumType, exp: EnumType) if dat.factory == exp.factory =>
          new TestableEnum(dat).expect(exp, message, format)
        case (dat: Record, exp: Record) =>
          new TestableRecord(dat).expect(exp, message, format)
        case (dat: Vec[_], exp: Vec[_]) if dat.getClass == exp.getClass =>
          new TestableVec(dat).expect(exp.asInstanceOf[dat.type], message, format)
        case (dat, exp) => throw new Exception(s"Don't know how to expect $exp from $dat")
      }
    }

    def poke(literal: T): Unit = (data, literal) match {
      case (x: UInt, lit: UInt) => new TestableUInt(x).poke(lit)
      case (x: SInt, lit: SInt) => new TestableSInt(x).poke(lit)
      case (x: EnumType, lit: EnumType) if x.factory == lit.factory =>
        new TestableEnum(x).poke(lit)
      case (x: Record, lit: Record) => new TestableRecord(x).poke(lit)
      case (x: Vec[_], lit: Vec[_]) if x.getClass == lit.getClass =>
        new TestableVec(x).poke(lit.asInstanceOf[x.type])
      case (x, lit) => throw new Exception(s"Don't know how to poke $x with $lit")
    }
  }
}

trait PeekPokeAPI {

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
