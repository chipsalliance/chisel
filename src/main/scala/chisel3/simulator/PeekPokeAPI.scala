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
          case x => throw new Exception(s"encode Bool with value $x, not 0 or 1")
        }
      } else {
        throw new Exception(s"encode Bool with value $value, not 0 or 1")
      }
    }

    def peekBoolean(): Boolean = peek().litToBoolean

    override def expect(expected: Bool)(implicit sourceInfo: SourceInfo): Unit = expect[Bool](
      expected,
      (obs: Simulation.Value, exp: Bool) => obs.asBigInt == exp.litValue,
      formatObserved = (obs: Simulation.Value) => obs.asBigInt.toString,
      formatExpected = (exp: Bool) => exp.litValue.toString,
      sourceInfo = sourceInfo
    )

    def expect(value: Boolean)(implicit sourceInfo: SourceInfo): Unit = expect(value.B)

    def poke(value: Boolean): Unit = poke(value.B)
  }

  implicit final class TestableReset(val data: Reset) extends TestableElement[Reset] {
    def encode(width: Int, value: BigInt): Reset = TestableBool(data.asBool).encode(width, value)

    def poke(value: Boolean): Unit = poke(value.B)
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

    def expect(expected: T, buildMessage: (T, T, String) => String, allowPartial: Boolean)(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      require(DataMirror.checkTypeEquivalence(data, expected), "Type mismatch")

      // Not peeking the value beforehand or using `def` or `lazy val` results in a mysterious infinite recursion and StackOverflowError
      // The value is not used if the expected value matches and incurs an overhead
      // TODO: dig deeper amd see if we can avoid this
      val peekedValue = data.peek()

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

    override def poke(literal: T): Unit = data.elements.foreach { case (name, dataEl) =>
      val valueEl = literal.elements(name)
      require(
        dataEl.getClass == valueEl.getClass,
        s"Type mismatch for Record element '$name': expected ${dataEl.getClass}, got ${valueEl.getClass}"
      )
      dataEl.poke(valueEl)
    }
  }

  implicit class TestableVec[T <: Data](val data: Vec[T]) extends PeekPokable[Vec[T]] {
    override def peek(): Vec[T] = {
      val elementValues = data.getElements.map(_.peek().asInstanceOf[T])
      chiselTypeOf(data).Lit(elementValues.zipWithIndex.map { _.swap }: _*)
    }

    // internal
    private[simulator] final def _expect[U <: Data](expected: Vec[U], buildMessage: (Vec[T], Vec[U], Int) => String)(
      implicit sourceInfo: SourceInfo
    ): Unit = {
      require(
        expected.length == data.length,
        s"Vec length mismatch: Data port has ${data.length} elements while the expected value is of length ${expected.length}"
      )

      // Not peeking the value beforehand or using `def` or `lazy val` results in a mysterious infinite recursion and StackOverflowError
      // The value is not used if the expected value matches and incurs an overhead
      // TODO: dig deeper amd see if we can avoid this
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

    private[simulator] def _expect[U <: Data](
      expected:               Vec[U],
      buildMessage:           (Vec[T], Vec[U]) => String,
      appendFailedIndexToMsg: Boolean
    )(
      implicit sourceInfo: SourceInfo
    ): Unit =
      _expect[U](
        expected,
        (observed: Vec[T], expected: Vec[U], idx: Int) =>
          buildMessage(observed, expected) + (if (appendFailedIndexToMsg) s"; First mismatch at index $idx" else "")
      )

    override def expect(expected: Vec[T], buildMessage: (Vec[T], Vec[T]) => String)(
      implicit sourceInfo: SourceInfo
    ): Unit =
      _expect[T](
        expected,
        buildMessage,
        appendFailedIndexToMsg = true
      )

    // for internal use
    private[simulator] final def _poke[U <: Data](literal: Vec[U]): Unit = {
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

    override def poke(literal: Vec[T]): Unit = _poke[T](literal)
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

    def peek(): T = toPeekable.peek().asInstanceOf[T]

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
          new TestableVec(dat)._expect(exp, buildMsgFn _, appendFailedIndexToMsg = true)
        case (dat, exp) => throw new Exception(s"Don't know how to expect $exp from $dat")
      }
    }

    def poke(literal: T): Unit = (data, literal) match {
      case (x: Bool, lit: Bool)         => new TestableBool(x).poke(lit)
      case (x: UInt, lit: UInt)         => new TestableUInt(x).poke(lit)
      case (x: SInt, lit: SInt)         => new TestableSInt(x).poke(lit)
      case (x: EnumType, lit: EnumType) => new TestableEnum(x).poke(lit)
      case (x: Record, lit: Record)     => new TestableRecord(x).poke(lit)
      case (x: Vec[_], lit: Vec[_])     => new TestableVec(x)._poke(lit)
      case (x, lit)                     => throw new Exception(s"Don't know how to poke $x with $lit")
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
