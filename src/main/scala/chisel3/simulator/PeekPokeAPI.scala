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

trait Peekable[T <: Data] extends Peekable$Intf[T] {

  private[chisel3] def _peekImpl()(implicit sourceInfo: SourceInfo): T

  private[chisel3] def _expectImpl(expected: T)(implicit sourceInfo: SourceInfo): Unit = _expectImpl(expected, "")

  private[chisel3] def _expectImpl(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit

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

sealed trait TestableAggregate[T <: Aggregate] extends PeekPokable[T] with TestableAggregate$Intf[T] {

  private[chisel3] def _expectPartialImpl(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit

  private[chisel3] def _expectPartialImpl(expected: T)(implicit sourceInfo: SourceInfo): Unit =
    _expectPartialImpl(expected, "")
}

sealed trait TestableElement[T <: Element] extends PeekPokable[T] with TestableElement$Intf[T] {
  protected def isSigned = false

  private[simulator] def encode(width: Int, value: BigInt): T

  private[simulator] final def encode(value: Simulation.Value): T = {
    encode(value.bitCount, value.asBigInt)
  }

  final def peekValue(): Simulation.Value = {
    simulatedModule.willPeek()
    simulationPort.get(isSigned = isSigned)
  }

  private[chisel3] def _peekImpl()(implicit sourceInfo: SourceInfo): T = encode(peekValue())

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

  private[chisel3] def _expectWithBuildMessageImpl(expected: T, buildMessage: (T, T) => String)(
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

  private[chisel3] override def _expectImpl(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit =
    _expectWithBuildMessageImpl(expected, (_: T, _: T) => message)

  private[chisel3] def _expectBigIntImpl(expected: BigInt)(implicit sourceInfo: SourceInfo): Unit = expect[BigInt](
    expected,
    (obs: Simulation.Value, exp: BigInt) => obs.asBigInt == exp,
    formatObserved = (obs: Simulation.Value) => obs.asBigInt.toString,
    formatExpected = (exp: BigInt) => exp.toString,
    sourceInfo = sourceInfo
  )

  private[chisel3] def _expectBigIntImpl(expected: BigInt, message: String)(implicit sourceInfo: SourceInfo): Unit =
    expect[BigInt](
      expected,
      (obs: Simulation.Value, exp: BigInt) => obs.asBigInt == exp,
      buildMessage = (_: Simulation.Value, _: BigInt) => message,
      sourceInfo = sourceInfo
    )

  private[chisel3] override def _expectImpl(expected: T)(implicit sourceInfo: SourceInfo): Unit = {
    require(expected.isLit, s"Expected value: $expected must be a literal")
    expect(
      expected,
      (observed: Simulation.Value, expected: T) => observed.asBigInt == expected.litValue,
      formatObserved = (obs: Simulation.Value) => encode(obs).toString,
      formatExpected = (exp: T) => exp.toString,
      sourceInfo = sourceInfo
    )
  }
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

  implicit final class TestableBool(val data: Bool) extends TestableElement[Bool] with TestableBool$Intf {
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

    private[chisel3] override def _expectImpl(expected: Bool)(implicit sourceInfo: SourceInfo): Unit = expect[Bool](
      expected,
      (obs: Simulation.Value, exp: Bool) => obs.asBigInt == exp.litValue,
      formatObserved = (obs: Simulation.Value) => obs.asBigInt.toString,
      formatExpected = (exp: Bool) => exp.litValue.toString,
      sourceInfo = sourceInfo
    )

    private[chisel3] def _expectBooleanImpl(value: Boolean)(implicit sourceInfo: SourceInfo): Unit = _expectImpl(
      value.B
    )

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
    private[chisel3] override def _peekImpl()(implicit sourceInfo: SourceInfo): T = {
      chiselTypeOf(data)._makeLit(
        data.elements.toSeq.map { case (name: String, elt: Data) =>
          (rec: T) => rec.elements(name) -> elt.peek()
        }: _*
      )
    }

    private[chisel3] def _expectRecordImpl(
      expected:     T,
      buildMessage: (T, T, String) => String,
      allowPartial: Boolean = false
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
                rec.expectPartial(expEl.asInstanceOf[rec.type], message)
              case (true, vec: Vec[_]) =>
                vec.expectPartial(expEl.asInstanceOf[vec.type], message)
              case _ =>
                portEl.expect(expEl, message)
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

    private[chisel3] override def _expectPartialImpl(expected: T, message: String)(
      implicit sourceInfo: SourceInfo
    ): Unit =
      _expectRecordImpl(expected, defaultMessageBuilder(_, _, _, message), allowPartial = true)

    private[chisel3] override def _expectImpl(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit =
      _expectRecordImpl(expected, defaultMessageBuilder(_, _, _, message), allowPartial = false)

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
    private[chisel3] override def _peekImpl()(implicit sourceInfo: SourceInfo): Vec[T] = {
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

    private[chisel3] override def _expectPartialImpl(expected: Vec[T], message: String)(
      implicit sourceInfo: SourceInfo
    ): Unit =
      _expectVecImpl(expected, defaultMessageBuilder(_, _, _, message), allowPartial = true)

    private[chisel3] override def _expectImpl(expected: Vec[T], message: String)(
      implicit sourceInfo: SourceInfo
    ): Unit =
      _expectVecImpl(expected, defaultMessageBuilder(_, _, _, message), allowPartial = false)

    private[chisel3] def _expectVecImpl(
      expected:     Vec[T],
      buildMessage: (Vec[T], Vec[T], Int) => String,
      allowPartial: Boolean = false
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
              rec.expectPartial(expEl.asInstanceOf[rec.type], message)
            case (true, vec: Vec[_]) =>
              vec.expectPartial(expEl.asInstanceOf[vec.type], message)
            case _ =>
              datEl.expect(expEl, message)
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

    private[chisel3] def _peekImpl()(implicit sourceInfo: SourceInfo): T = toPeekable.peek().asInstanceOf[T]

    private[chisel3] override def _expectImpl(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit = {
      (data, expected) match {
        case (dat: Bool, exp: Bool) =>
          new TestableBool(dat).expect(exp, message)
        case (dat: UInt, exp: UInt) =>
          new TestableUInt(dat).expect(exp, message)
        case (dat: SInt, exp: SInt) =>
          new TestableSInt(dat).expect(exp, message)
        case (dat: EnumType, exp: EnumType) if dat.factory == exp.factory =>
          new TestableEnum(dat).expect(exp, message)
        case (dat: Record, exp: Record) =>
          new TestableRecord(dat).expect(exp, message)
        case (dat: Vec[_], exp: Vec[_]) if dat.getClass == exp.getClass =>
          new TestableVec(dat).expect(exp.asInstanceOf[dat.type], message)
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
