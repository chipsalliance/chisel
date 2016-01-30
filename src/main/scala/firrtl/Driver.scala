package firrtl

import java.io._
import java.nio.file.{Paths, Files}

import scala.io.Source
import scala.sys.process._

import com.typesafe.scalalogging.LazyLogging

import Utils._
import DebugUtils._
import Passes._

object Driver extends LazyLogging {
  private val usage = """
    Usage: java -cp utils/bin/firrtl.jar firrtl.Driver [options] -i <input> -o <output>
  """
  private val defaultOptions = Map[Symbol, Any]().withDefaultValue(false)

  private def compile(input: String, output: String, compiler: Compiler)
  {
    val parsedInput = Parser.parse(input, Source.fromFile(input).getLines)
    val writerOutput = new PrintWriter(new File(output))
    compiler.run(parsedInput, writerOutput)
  }

  def main(args: Array[String])
  {
    val arglist = args.toList
    type OptionMap = Map[Symbol, Any]

    // Default debug mode is 'debug
    def decodeDebugMode(mode: Any): Symbol =
      mode match {
        case s: String => Symbol(s)
        case _ => 'debug
      }

    def nextPrintVar(syms: List[Symbol], chars: List[Char]): List[Symbol] =
      chars match {
        case Nil => syms
        case 't' :: tail => nextPrintVar(syms ++ List('types), tail)
        case 'k' :: tail => nextPrintVar(syms ++ List('kinds), tail)
        case 'w' :: tail => nextPrintVar(syms ++ List('widths), tail)
        case 'T' :: tail => nextPrintVar(syms ++ List('twidths), tail)
        case 'g' :: tail => nextPrintVar(syms ++ List('genders), tail)
        case 'c' :: tail => nextPrintVar(syms ++ List('circuit), tail)
        case 'd' :: tail => nextPrintVar(syms ++ List('debug), tail)
        case 'i' :: tail => nextPrintVar(syms ++ List('info), tail)
        case char :: tail => throw new Exception("Unknown print option " + char)
      }

    def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      list match {
        case Nil => map
        case "-X" :: value :: tail =>
                  nextOption(map ++ Map('compiler -> value), tail)
        case "-p" :: value :: tail =>
                  nextOption(map ++ Map('printVars -> value), tail)
        case "-i" :: value :: tail =>
                  nextOption(map ++ Map('input -> value), tail)
        case "-o" :: value :: tail =>
                  nextOption(map ++ Map('output -> value), tail)
        case ("-h" | "--help") :: tail =>
                  nextOption(map ++ Map('help -> true), tail)
        case option :: tail =>
                  throw new Exception("Unknown option " + option)
      }
    }
    val options = nextOption(defaultOptions, arglist)

    if (options('help) == true) {
      println(usage)
      System.exit(0)
    }

    val input = options('input) match {
      case s: String => s
      case false => throw new Exception("No input file provided!" + usage)
    }
    val output = options('output) match {
      case s: String => s
      case false => throw new Exception("No output file provided!" + usage)
    }
    val printVars = options('printVars) match {
      case s: String => nextPrintVar(List(), s.toList)
      case false => List()
    }

    if (!printVars.isEmpty) {
      logger.warn("-p options currently ignored")
      if (!logger.underlying.isDebugEnabled) {
        logger.warn("-p options will only print at DEBUG log level, logging configuration can be edited in src/main/resources/logback.xml")
      }
    }

    options('compiler) match {
      case "verilog" => compile(input, output, VerilogCompiler)
      case "firrtl" => compile(input, output, FIRRTLCompiler)
      case other => throw new Exception("Invalid compiler! " + other)
    }
  }

  def time[R](str: String)(block: => R): R = {
    val t0 = System.currentTimeMillis()
    val result = block    // call-by-name
    val t1 = System.currentTimeMillis()
    logger.info(s"Time to ${str}: ${t1 - t0} ms")
    result
  }
}
