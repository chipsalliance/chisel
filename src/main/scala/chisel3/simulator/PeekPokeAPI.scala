package chisel3.simulator

import svsim._
import chisel3._

import chisel3.experimental.{SourceInfo, SourceLine}
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

  implicit class testableClock(clock: Clock) {
    def step(cycles: Int = 1, period: Int = 10): Unit = {
      require(
        period >= 2,
        s"specified period, '${period}', must be 2 or greater because an integer half period must be non-zero"
      )

      val module = AnySimulatedModule.current
      module.willEvaluate()
      if (cycles == 0) {
        module.controller.run(0)
      } else {
        val simulationPort = module.port(clock)
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

      val module = AnySimulatedModule.current
      module.willEvaluate()
      val simulationPort = module.port(clock)
      simulationPort.tick(
        timestepsPerPhase = period / 2,
        maxCycles = maxCycles,
        inPhaseValue = 0,
        outOfPhaseValue = 1,
        sentinel = Some(module.port(sentinelPort), sentinelValue)
      )
    }
  }

  sealed trait SimulationData[T <: Data] {
    val data: T

    private def isSigned = data.isInstanceOf[SInt]

    private[simulator] def encode(width: Int, value: BigInt): T
    private final def encode(value: Simulation.Value): T = {
      encode(value.bitCount, value.asBigInt)
    }

    final def peek(): T = encode(data.peekValue())
    final def expect(expected: T)(implicit sourceInfo: SourceInfo): Unit = {
      data.expect(
        expected.litValue,
        encode(_).litValue,
        (observed: BigInt, expected: BigInt) => s"Expectation failed: observed value $observed != $expected",
        sourceInfo
      )
    }
    final def expect(expected: T, message: String)(implicit sourceInfo: SourceInfo): Unit = {
      data.expect(expected.litValue, encode(_).litValue, (_: BigInt, _: BigInt) => message, sourceInfo)
    }
    final def expect(expected: BigInt)(implicit sourceInfo: SourceInfo): Unit = {
      data.expect(
        expected,
        _.asBigInt,
        (observed: BigInt, expected: BigInt) => s"Expectation failed: observed value $observed != $expected",
        sourceInfo
      )
    }
    final def expect(expected: BigInt, message: String)(implicit sourceInfo: SourceInfo): Unit = {
      data.expect(expected, _.asBigInt, (_: BigInt, _: BigInt) => message, sourceInfo)
    }

  }

  implicit final class testableSInt(val data: SInt) extends SimulationData[SInt] {
    override def encode(width: Int, value: BigInt) = value.asSInt(width.W)
  }

  implicit final class testableUInt(val data: UInt) extends SimulationData[UInt] {
    override def encode(width: Int, value: BigInt) = value.asUInt(width.W)
  }

  implicit final class testableBool(val data: Bool) extends SimulationData[Bool] {
    override def encode(width: Int, value: BigInt): Bool = {
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
  }

  implicit final class testableData[T <: Data](data: T) {
    private def isSigned = data.isInstanceOf[SInt]

    def poke(boolean: Boolean): Unit = {
      poke(if (boolean) 1 else 0)
    }
    def poke(literal: T): Unit = {
      poke(literal.litValue)
    }
    def poke(value: BigInt): Unit = {
      val module = AnySimulatedModule.current
      module.willPoke()
      val simulationPort = module.port(data)
      simulationPort.set(value)
    }
    def peekValue(): Simulation.Value = {
      val module = AnySimulatedModule.current
      module.willPeek()
      val simulationPort = module.port(data)
      simulationPort.get(isSigned = isSigned)
    }
    def expect[T](
      expected:     T,
      encode:       (Simulation.Value) => T,
      buildMessage: (T, T) => String,
      sourceInfo:   SourceInfo
    ): Unit = {
      val module = AnySimulatedModule.current
      module.willPeek()
      val simulationPort = module.port(data)

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
}
