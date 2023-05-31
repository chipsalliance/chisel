// SPDX-License-Identifier: Apache-2.0

package svsim

import scala.collection.mutable.Queue
import scala.util.Try
import java.io.{BufferedReader, BufferedWriter, File, InputStreamReader, OutputStreamWriter}

final class Simulation private[svsim] (
  executableName:           String,
  settings:                 Simulation.Settings,
  val workingDirectoryPath: String,
  moduleInfo:               ModuleInfo) {
  private val executionScriptPath = s"$workingDirectoryPath/execution-script.txt"

  def run[T](body: Simulation.Controller => T): T = run()(body)
  def run[T](
    conservativeCommandResolution: Boolean = false,
    verbose:                       Boolean = false,
    executionScriptLimit:          Option[Int] = None
  )(body:                          Simulation.Controller => T
  ): T = {
    val cwd = settings.customWorkingDirectory match {
      case None => workingDirectoryPath
      case Some(value) =>
        if (value.startsWith("/"))
          value
        else
          s"$workingDirectoryPath/$value"
    }
    val command = Seq(s"$workingDirectoryPath/$executableName") ++ settings.arguments
    val processBuilder = new ProcessBuilder(command: _*)
    processBuilder.directory(new File(cwd))
    processBuilder.redirectError(ProcessBuilder.Redirect.INHERIT)
    val environment = settings.environment ++ Seq(
      Some("SVSIM_EXECUTION_SCRIPT" -> executionScriptPath),
      executionScriptLimit.map("SVSIM_EXECUTION_SCRIPT_LIMIT" -> _.toString)
    ).flatten
    environment.foreach { (pair) =>
      processBuilder.environment().put(pair._1, pair._2)
    }
    val process = processBuilder.start()
    val controller = new Simulation.Controller(
      new BufferedWriter(new OutputStreamWriter(process.getOutputStream())),
      new BufferedReader(new InputStreamReader(process.getInputStream())),
      moduleInfo,
      conservativeCommandResolution = conservativeCommandResolution,
      logMessagesAndCommands = verbose
    )
    val bodyOutcome = Try {
      val result = body(controller)
      // Exceptions thrown from commands still in the queue when `body` returns should supercede returning `result`
      controller.completeInFlightCommands()
      result
    }

    // Always attempt graceful shutdown, even if `body` failed.
    val gracefulShutdownOutcome = Try[Unit] {
      // If the process is still running, give it an opportunity to shut down gracefully
      if (process.isAlive()) {
        controller.sendCommand(Simulation.Command.Done)
        controller.completeInFlightCommands()
        process.waitFor()
      }
    }

    // Ensure process is destroyed prior to returning or throwing an exception
    process.destroyForcibly()

    // Exceptions thrown from `body` have the highest priority
    val result = bodyOutcome.get

    // Nonzero exit status supercedes graceful-shutdown exceptions
    if (process.exitValue() != 0) {
      throw new Exception(s"Nonzero exit status: ${process.exitValue()}")
    }

    // Issues during graceful shutdown are considered test failures
    gracefulShutdownOutcome.get

    result
  }

}
object Simulation {
  private[svsim] final case class Settings(
    customWorkingDirectory: Option[String] = None,
    arguments:              Seq[String] = Seq(),
    environment:            Map[String, String] = Map())

  /** @note Methods in this class and `Simulation.Port` are somewhat lazy in their execution. Specifically, methods returning `Unit` neither flush the command buffer, nor do they actively read from the message buffer. Only commands which return a value will wait to return until the simulation has progressed to the point where the value is available. This can improve performance by essentially enabling batching of both commands and messages. If you want to ensure that all commands have been sent to the simulation executable, you can call `completeInFlightCommands()`.
    */
  final class Controller private[Simulation] (
    commandWriter:                 BufferedWriter,
    messageReader:                 BufferedReader,
    moduleInfo:                    ModuleInfo,
    conservativeCommandResolution: Boolean = false,
    logMessagesAndCommands:        Boolean = false) {

    private def readStringOfLength(length: Int): String = {
      val array = new Array[Char](length)
      val readLength = messageReader.read(array)
      if (readLength != length) {
        throw new Exception(s"Expected string of length ${length} but got ${readLength}")
      }
      new String(array)
    }

    private var readMessageCount = 0
    // For specific message formats, consult `simulation-driver.cpp`
    private def readNextAvailableMessage(): Simulation.Message = {
      object MessageCode {
        val Ready = 'r'
        val Error = 'e'
        val Ack = 'k'
        val Bits = 'b'
        val Log = 'l'
      }

      def readChar(): Option[Char] = {
        val array = new Array[Char](1)
        val read = messageReader.read(array)
        if (read != 1) {
          None
        } else {
          Some(array(0))
        }
      }

      val messageCode = readChar() match {
        case Some(c) => c
        case None    => throw Simulation.UnexpectedEndOfMessages
      }

      def mustRead(char: Char) = {
        val nextChar = readChar()
        if (nextChar != Some(char)) {
          throw new Exception(s"Expected '${char}' but got '${messageReader.readLine()}'")
        }
      }
      mustRead(' ')

      import Simulation.Message._
      val message: Message = messageCode match {
        case MessageCode.Ready => {
          val rest = messageReader.readLine()
          if (rest != "ready") {
            throw new Exception(s"Malformed ready message: ${messageCode} ${rest}")
          }
          Ready
        }
        case MessageCode.Error => {
          throw new Error(messageReader.readLine())
        }
        case MessageCode.Ack => {
          val rest = messageReader.readLine()
          if (rest != "ack") {
            throw new Exception(s"Malformed ack message: ${messageCode} ${rest}")
          }
          Ack
        }
        case MessageCode.Bits => {
          val bitCount = Integer.parseInt(readStringOfLength(8), 16)
          mustRead(' ')
          Bits(bitCount, BigInt(messageReader.readLine(), 16))
        }
        case MessageCode.Log => {
          val length = Integer.parseInt(readStringOfLength(8), 16)
          if (length < 0) {
            throw new Exception("Invalid log message length")
          }
          mustRead(' ')
          val content = readStringOfLength(length)
          mustRead('\n')
          Log(new String(content))
        }
        case _ => throw new Exception(s"Unknown message code: ${messageCode}")
      }
      if (logMessagesAndCommands) {
        // NOTE: Commands are 1-indexed, but messages are 0-indexed since we read the first message (READY) before we send any commands.
        println(s"Received message ${readMessageCount}: ${message}")
      }
      readMessageCount += 1
      message
    }

    private val expectations: Queue[PartialFunction[Simulation.Message, Unit]] = Queue.empty

    def completeInFlightCommands() = {
      commandWriter.flush()

      expectations.foreach { f =>
        val message = readNextAvailableMessage()
        if (f.isDefinedAt(message)) {
          f(message)
        } else {
          throw new Exception(s"Unexpected message: ${message}")
        }
      }
      expectations.clear()
    }

    private[Simulation] def processNextMessage[A](f: PartialFunction[Simulation.Message, A]): A = {
      completeInFlightCommands()

      val message = readNextAvailableMessage()
      if (f.isDefinedAt(message)) {
        f(message)
      } else {
        throw new Exception(s"Unexpected message: ${message}")
      }
    }

    private[Simulation] def expectNextMessage(f: PartialFunction[Simulation.Message, Unit]) = {
      if (conservativeCommandResolution) {
        processNextMessage(f)
      } else {
        expectations.enqueue(f)
      }
    }

    private var sentCommandCount = 0
    private[Simulation] def sendCommand(command: Simulation.Command) = {
      object CommandCode extends Enumeration {
        val Done = 'D'
        val Log = 'L'
        val GetBits = 'G'
        val SetBits = 'S'
        val Run = 'R'
        val Tick = 'T'
        val Trace = 'W'
      };

      sentCommandCount += 1
      if (logMessagesAndCommands) {
        // NOTE: Commands are 1-indexed, but messages are 0-indexed since we read the first message (READY) before we send any commands.
        println(s"Sending command ${sentCommandCount}: ${command}")
      }

      import Simulation.Command._
      // For specific command formats, consult `simulation-driver.cpp`
      command match {
        case Done => {
          commandWriter.write(CommandCode.Done)
        }
        case Log => {
          commandWriter.write(CommandCode.Log)
        }
        case GetBits(id, isSigned) => {
          commandWriter.write(CommandCode.GetBits)
          commandWriter.write(" ")
          commandWriter.write(if (isSigned) "s" else "u")
          commandWriter.write(" ")
          commandWriter.write(id)
        }
        case SetBits(id, value) => {
          commandWriter.write(CommandCode.SetBits)
          commandWriter.write(" ")
          commandWriter.write(id)
          commandWriter.write(" ")
          commandWriter.write(value.toString(16))
        }
        case Run(timesteps) => {
          commandWriter.write(CommandCode.Run)
          commandWriter.write(" ")
          commandWriter.write(timesteps.toHexString)
        }
        case Tick(id, inPhaseValue, outOfPhaseValue, timestepsPerPhase, maxCycles, sentinel) => {
          commandWriter.write(CommandCode.Tick)
          commandWriter.write(" ")
          commandWriter.write(id)
          commandWriter.write(" ")
          commandWriter.write(inPhaseValue.toString(16))
          commandWriter.write(",")
          commandWriter.write(outOfPhaseValue.toString(16))
          commandWriter.write("-")
          commandWriter.write(timestepsPerPhase.toHexString)
          commandWriter.write("*")
          commandWriter.write(maxCycles.toHexString)
          sentinel match {
            case Some((port, value)) => {
              commandWriter.write(" ")
              commandWriter.write(port.id)
              commandWriter.write("=")
              commandWriter.write(value.toString(16))
            }
            case None =>
          }
        }
        case Trace(enable) => {
          commandWriter.write(CommandCode.Trace)
          commandWriter.write(" ")
          commandWriter.write(if (enable) "1" else "0")
        }
      }
      commandWriter.newLine()
    }

    def readLog(): String = {
      sendCommand(Simulation.Command.Log)
      processNextMessage {
        case Simulation.Message.Log(message) =>
          message
      }
    }

    def run(timesteps: Int) = {
      sendCommand(Simulation.Command.Run(timesteps))
      expectNextMessage { case Simulation.Message.Ack => }
    }

    def setTraceEnabled(enabled: Boolean): Unit = {
      sendCommand(Simulation.Command.Trace(enabled))
      expectNextMessage { case Simulation.Message.Ack => }
    }

    private val portInfos = moduleInfo.ports.zipWithIndex.map {
      case (port, index) =>
        port.name -> (index.toHexString, port)
    }.toMap

    def port(name: String): Simulation.Port = {
      val (id, info) = portInfos.get(name).get
      new Simulation.Port(this, id, info)
    }

    // Simulation starts with a single `READY` message
    expectNextMessage { case Simulation.Message.Ready => }
  }

  sealed trait Message
  object Message {
    case object Ready extends Message
    case object Ack extends Message
    case class Error(message: String) extends Throwable(message) with Message
    case class Bits(count: Int, value: BigInt) extends Message
    case class Log(message: String) extends Message
  }

  case object UnexpectedEndOfMessages extends Exception

  sealed trait Command
  object Command {
    case object Done extends Command
    case object Log extends Command
    case class GetBits(id: String, isSigned: Boolean) extends Command
    case class SetBits(id: String, value: BigInt) extends Command
    case class Run(timesteps: Int) extends Command
    case class Tick(
      id:                String,
      inPhaseValue:      BigInt,
      outOfPhaseValue:   BigInt,
      timestepsPerPhase: Int,
      maxCycles:         Int,
      sentinel:          Option[(Port, BigInt)])
        extends Command
    case class Trace(enable: Boolean) extends Command
  }

  final case class Value(bitCount: Int, asBigInt: BigInt)

  final case class Port private[Simulation] (controller: Simulation.Controller, id: String, info: ModuleInfo.Port) {

    def set(value: BigInt) = {
      controller.sendCommand(Simulation.Command.SetBits(id, value))
      controller.expectNextMessage {
        case Simulation.Message.Ack =>
      }
    }

    def get(isSigned: Boolean = false): Value = {
      controller.sendCommand(Simulation.Command.GetBits(id, isSigned))
      controller.processNextMessage {
        case Simulation.Message.Bits(bitCount, value) =>
          Value(bitCount, value)
      }
    }

    def tick(timestepsPerPhase: Int, cycles: Int, inPhaseValue: BigInt, outOfPhaseValue: BigInt) = {
      controller.sendCommand(
        Simulation.Command.Tick(id, inPhaseValue, outOfPhaseValue, timestepsPerPhase, cycles, None)
      )
      controller.expectNextMessage {
        case Simulation.Message.Bits(_, _) =>
      }
    }

    def tick(
      timestepsPerPhase: Int,
      maxCycles:         Int,
      inPhaseValue:      BigInt,
      outOfPhaseValue:   BigInt,
      sentinel:          Option[(Port, BigInt)]
    ): BigInt = {
      controller.sendCommand(
        Simulation.Command.Tick(id, inPhaseValue, outOfPhaseValue, timestepsPerPhase, maxCycles, sentinel)
      )
      controller.processNextMessage {
        case Simulation.Message.Bits(_, cyclesElapsed) =>
          cyclesElapsed
      }
    }

    def tick(
      timestepsPerPhase:      Int,
      maxCycles:              Int,
      inPhaseValue:           BigInt,
      outOfPhaseValue:        BigInt,
      sentinel:               Option[(Port, BigInt)],
      checkElapsedCycleCount: (BigInt) => Unit
    ) = {
      controller.sendCommand(
        Simulation.Command.Tick(id, inPhaseValue, outOfPhaseValue, timestepsPerPhase, maxCycles, sentinel)
      )
      controller.expectNextMessage {
        case Simulation.Message.Bits(_, cyclesElapsed) =>
          checkElapsedCycleCount(cyclesElapsed)
      }
    }

    def check(f: Value => Unit): Unit = check()(f)
    def check(isSigned: Boolean = false)(f: Value => Unit): Unit = {
      controller.sendCommand(Simulation.Command.GetBits(id, isSigned))
      controller.expectNextMessage {
        case Simulation.Message.Bits(bitCount, value) =>
          f(Value(bitCount, value))
      }
    }
  }
}
