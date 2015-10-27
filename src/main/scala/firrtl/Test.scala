package firrtl

import java.io._
import Utils._
import DebugUtils._
import Passes._

object Test
{
  private val usage = """
    Usage: java -cp utils/bin/firrtl.jar firrtl.Test [options] -i <input> -o <output>
  """
  private val defaultOptions = Map[Symbol, Any]().withDefaultValue(false)

  // Parse input file and print to output
  private def highFIRRTL(input: String, output: String)(implicit logger: Logger)
  {
    val ast = Parser.parse(input)
    val writer = new PrintWriter(new File(output))
    writer.write(ast.serialize())
    writer.close()
    logger.printlnDebug(ast)
  }
  private def verilog(input: String, output: String)(implicit logger: Logger)
  {
    logger.warn("Verilog compiler not fully implemented")
    val ast = time("parse"){ Parser.parse(input) }
    // Execute passes

    logger.println("Infer Types")
    val ast2 = time("inferTypes"){ inferTypes(ast) }
    logger.printlnDebug(ast2)
    logger.println("Finished Infer Types")
    //val ast2 = ast

    // Output
    val writer = new PrintWriter(new File(output))
    var outString = time("serialize"){ ast2.serialize() }
    writer.write(outString)
    writer.close()
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
        case 'd' :: tail => nextPrintVar(syms ++ List('debug), tail) // Currently ignored
        case 'i' :: tail => nextPrintVar(syms ++ List('info), tail)  
        case char :: tail => throw new Exception("Unknown print option " + char)
      }

    def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      list match {
        case Nil => map
        case "-X" :: value :: tail => 
                  nextOption(map ++ Map('compiler -> value), tail)
        case "-d" :: value :: tail => 
                  nextOption(map ++ Map('debugMode -> value), tail)
        case "-l" :: value :: tail =>
                  nextOption(map ++ Map('log -> value), tail)
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
    val debugMode = decodeDebugMode(options('debugMode))
    val printVars = options('printVars) match {
      case s: String => nextPrintVar(List(), s.toList)
      case false => List()
    }
    implicit val logger = options('log) match {
      case s: String => Logger(new PrintWriter(new FileOutputStream(s)), debugMode, printVars)
      case false => Logger(new PrintWriter(System.err, true), debugMode, printVars)
    }

    // -p "printVars" options only print for debugMode > 'debug, warn if -p enabled and debugMode < 'debug
    if( !logger.debugEnable && !printVars.isEmpty )
      logger.warn("-p options will not print unless debugMode (-d) is debug or trace")

    options('compiler) match {
      case "verilog" => verilog(input, output)
      case "HighFIRRTL" => highFIRRTL(input, output)
      case other => throw new Exception("Invalid compiler! " + other)
    }
  }

  def time[R](str: String)(block: => R)(implicit logger: Logger): R = {
    val t0 = System.currentTimeMillis()
    val result = block    // call-by-name
    val t1 = System.currentTimeMillis()
    logger.info(s"Time to ${str}: ${t1 - t0} ms")
    result
  }
}
