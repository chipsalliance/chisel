package firrtl

import java.io._
import Utils._

object Test
{
  private val usage = """
    Usage: java -jar firrtl.jar firrtl.Test [options] -i <input> -o <output>
  """
  private val defaultOptions = Map[Symbol, Any]().withDefaultValue(false)

  // Parse input file and print to output
  private def highFIRRTL(input: String, output: String)
  {
    val ast = Parser.parse(input)
    val writer = new PrintWriter(new File(output))
    writer.write(ast.serialize)
    writer.close()
  }

  def main(args: Array[String])
  {
    val arglist = args.toList
    type OptionMap = Map[Symbol, Any]

    def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      def isSwitch(s: String) = (s(0) == '-')
      list match {
        case Nil => map
        case "-X" :: value :: tail => 
                  nextOption(map ++ Map('compiler -> value), tail)
        //case "-d" :: tail => 
        //          nextOption(map ++ Map('debug -> true), tail)
        case "-i" :: value :: tail =>
                  nextOption(map ++ Map('input -> value), tail)
        case "-o" :: value :: tail =>
                  nextOption(map ++ Map('output -> value), tail)
        case option :: tail => 
                  throw new Exception("Unknown option " + option)
      }
    }
    val options = nextOption(defaultOptions, arglist)
    println(options)

    val input = options('input) match {
      case s: String => s
      case false => throw new Exception("No input file provided!" + usage)
    }
    val output = options('output) match {
      case s: String => s
      case false => throw new Exception("No output file provided!" + usage)
    }

    options('compiler) match {
      case "Verilog" => throw new Exception("Verilog compiler not currently supported!")
      case "HighFIRRTL" => highFIRRTL(input, output)
      case other => throw new Exception("Invalid compiler! " + other)
    }
  }
}
