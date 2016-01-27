package firrtl

import java.io._
import java.nio.file.{Paths, Files}

import scala.io.Source
import scala.sys.process._

import com.typesafe.scalalogging.LazyLogging

import Utils._
import DebugUtils._
import Passes._

trait DriverPass {
  def run(input: String, output: String) : Unit
}
case class StanzaPass(val passes : Seq[String]) extends DriverPass with LazyLogging {
  def run(input : String, output : String): Unit = {
    val cmd = Seq("firrtl-stanza", "-i", input, "-o", output, "-b", "firrtl") ++ passes.flatMap(x=>Seq("-x", x))
    logger.info(cmd.mkString(" "))
    val ret = cmd.!!
    logger.info(ret)
  }
}
case class ScalaPass(val func : Circuit => Circuit) extends DriverPass with LazyLogging {
  def run(input : String, output : String): Unit = {
    var ast = Parser.parse(input, Source.fromFile(input).getLines)
    val newast = func(ast)
    logger.info("Writing to " + output)
    val writer = new PrintWriter(new File(output))
    writer.write(newast.serialize())
    writer.close()
  }
}
object StanzaPass {
  def apply(pass: String): StanzaPass = StanzaPass(Seq(pass))
}

object DriverPasses {
  private def aggregateStanzaPasses(passes: Seq[DriverPass]): Seq[DriverPass] = {
    if (passes.isEmpty) return Seq()
    val span = passes.span(x => x match {
      case p : StanzaPass => true
      case _ => false
    })
    if (span._1.isEmpty) {
      Seq(span._2.head) ++ aggregateStanzaPasses(span._2.tail)
    } else {
      Seq(StanzaPass(span._1.flatMap(x=>x.asInstanceOf[StanzaPass].passes))) ++ aggregateStanzaPasses(span._2)
    }
  }

  def optimize(passes: Seq[DriverPass]): Seq[DriverPass] = {
    aggregateStanzaPasses(passes)
  }
}

object Driver extends LazyLogging {
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

  val defaultPasses = DriverPasses.optimize(Seq(
    StanzaPass("to-firrtl"),

    StanzaPass("high-form-check"),

//  ScalaPass(renameall(Map(
//    "c"->"ccc",
//    "z"->"zzz",
//    "top"->"its_a_top_module"
//  ))),
    // StanzaPass("temp-elim"), // performance pass
    StanzaPass("to-working-ir"),

    StanzaPass("resolve-kinds"),
    StanzaPass("infer-types"),
    StanzaPass("check-types"),
    StanzaPass("resolve-genders"),
    StanzaPass("check-genders"),
    StanzaPass("infer-widths"),
    StanzaPass("width-check"),

    StanzaPass("check-kinds"),

    StanzaPass("expand-accessors"),
    StanzaPass("lower-to-ground"),
    StanzaPass("inline-indexers"),
    StanzaPass("infer-types"),
    //ScalaPass(inferTypes),
    StanzaPass("check-genders"),
    StanzaPass("expand-whens"),

    StanzaPass("real-ir"),

    StanzaPass("pad-widths"),
    StanzaPass("const-prop"),
    StanzaPass("split-expressions"),
    StanzaPass("width-check"),
    StanzaPass("high-form-check"),
    StanzaPass("low-form-check"),
    StanzaPass("check-init")//,
    //ScalaPass(renamec)
  ))

  // Parse input file and print to output
  private def firrtl(input: String, output: String)
  {
    val ast = Parser.parse(input, Source.fromFile(input).getLines)
    val writer = new PrintWriter(new File(output))
    writer.write(ast.serialize())
    writer.close()
    logger.debug(ast.toString)
  }

  def executePasses(ast: Circuit, passes: Seq[Circuit => Circuit]): Circuit = {
    if (passes.isEmpty) ast
    else executePasses(passes.head(ast), passes.tail)
  }

  private def verilog(input: String, output: String) {
    val outfile = defaultPasses.foldLeft( input ) ( (infile, pass) => {
      val outfile = genTempFilename(output)
      pass.run(infile, outfile)
      outfile
    })

    logger.info(outfile)

    // finally, convert to verilog at the end
    val cmd = Seq("firrtl-stanza", "-i", outfile, "-o", output, "-X", "verilog")
    logger.info(cmd.mkString(" "))
    val ret = cmd.!!
    logger.info(ret)
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
      case "verilog" => verilog(input, output)
      case "firrtl" => firrtl(input, output)
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
