package firrtl

import java.io._
import scala.sys.process._
import java.nio.file.{Paths, Files}
import Utils._
import DebugUtils._
import Passes._

object Driver
{
  private val usage = """
    Usage: java -cp utils/bin/firrtl.jar firrtl.Driver [options] -i <input> -o <output>
  """
  private val defaultOptions = Map[Symbol, Any]().withDefaultValue(false)

  // Appends 0 to the filename and appends .tmp to the extension
  private def genTempFilename(filename: String): String = {
    val pat = """(.*/)([^/]*)([.][^/.]*)""".r 
    val (path, name, ext) = filename match {
      case pat(path, name, ext) => (path, name, ext + ".tmp")
      case _ => ("./", "temp", ".tmp")
    }
    var count = 0
    while( Files.exists(Paths.get(path + name + count + ext )) ) 
      count += 1
    path + name + count + ext
  }

  // Parse input file and print to output
  private def firrtl(input: String, output: String)(implicit logger: Logger)
  {
    val ast = Parser.parse(input)
    val writer = new PrintWriter(new File(output))
    writer.write(ast.serialize())
    writer.close()
    logger.printlnDebug(ast)
  }

  private def verilog(input: String, output: String, stanza: String)(implicit logger: Logger)
  {
    val stanzaPass = //List( 
      List("rem-spec-chars", "high-form-check",
      "temp-elim", "to-working-ir", "resolve-kinds", "infer-types",
      "resolve-genders", "check-genders", "check-kinds", "check-types",
      "expand-accessors", "lower-to-ground", "inline-indexers", "infer-types",
      "check-genders", "expand-whens", "infer-widths", "real-ir", "width-check",
      "pad-widths", "const-prop", "split-expressions", "width-check",
      "high-form-check", "low-form-check", "check-init")
    //)
    val scalaPass = List(List[String]())

    val mapString2Pass = Map[String, Circuit => Circuit] (
      "infer-types" -> inferTypes
    )

    if (stanza.isEmpty || !Files.exists(Paths.get(stanza)))
      throw new FileNotFoundException("Stanza binary not found! " + stanza)

    // For now, just use the stanza implementation in its entirety
    val cmd = Seq(stanza, "-i", input, "-o", output, "-b", "verilog") ++ stanzaPass.flatMap(Seq("-x", _))
    println(cmd.mkString(" "))
    val ret = cmd.!!
    println(ret)

    // Switch between stanza and scala implementations
    //var scala2Stanza = input
    //for ((stanzaPass, scalaPass) <- stanzaPass zip scalaPass) {
    //  val stanza2Scala = genTempFilename(output)
    //  val cmd: Seq[String] = Seq[String](stanza, "-i", scala2Stanza, "-o", stanza2Scala, "-b", "firrtl") ++ stanzaPass.flatMap(Seq("-x", _))
    //  println(cmd.mkString(" "))
    //  val ret = cmd.!!
    //  println(ret)

    //  if( scalaPass.isEmpty ) {
    //    scala2Stanza = stanza2Scala
    //  } else {
    //    var ast = Parser.parse(stanza2Scala) 
    //    //scalaPass.foreach( f => (ast = f(ast)) ) // Does this work?
    //    for ( f <- scalaPass ) yield { ast = mapString2Pass(f)(ast) }

    //    scala2Stanza = genTempFilename(output)
    //    val writer = new PrintWriter(new File(scala2Stanza))
    //    writer.write(ast.serialize())
    //    writer.close()
    //  }
    //}
    //val cmd = Seq(stanza, "-i", scala2Stanza, "-o", output, "-b", "verilog") 
    //println(cmd.mkString(" "))
    //val ret = cmd.!!
    //println(ret)
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
        case "--stanza-bin" :: value :: tail =>
                  nextOption(map ++ Map('stanza -> value), tail)
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
    val stanza = options('stanza) match {
      case s: String => s
      case false => ""
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
      case "verilog" => verilog(input, output, stanza)
      case "firrtl" => firrtl(input, output)
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
