/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

package firrtl

import java.io.{PrintWriter, Writer, File}
import scala.io.Source
import scala.collection.mutable

import Utils._
import Parser.{InfoMode, IgnoreInfo, UseInfo, GenInfo, AppendInfo}

object Driver {
  private val usage = """
Usage: sbt "run-main firrtl.Driver -i <input_file> -o <output_file> -X <compiler>"
       firrtl -i <input_file> -o <output_file> -X <compiler> [options]
Options:
  -X <compiler>         Specify the target compiler
                        Currently supported: high low verilog
  --info-mode <mode>    Specify Info Mode
                        Supported modes: ignore, use, gen, append
  """

  // Compiles circuit. First parses a circuit from an input file,
  //  executes all compiler passes, and writes result to an output
  //  file.
  def compile(
      input: String, 
      output: String, 
      compiler: Compiler, 
      infoMode: InfoMode = IgnoreInfo,
      annotations: Seq[CircuitAnnotation] = Seq.empty) = {
    val parsedInput = Parser.parse(Source.fromFile(input).getLines, infoMode)
    val writerOutput = new PrintWriter(new File(output))
    compiler.compile(parsedInput, annotations, writerOutput)
    writerOutput.close
  }

  // Arguments specify the compiler, input file, and output file
  def main(args: Array[String]) = {
    val arglist = args.toList

    sealed trait CompilerOption
    case object InputFileName extends CompilerOption
    case object OutputFileName extends CompilerOption
    case object CompilerName extends CompilerOption
    case object InfoModeOption extends CompilerOption
    val defaultOptions = Map[CompilerOption, String]()

    // Inline Annotation datastructure/function
    val inlineAnnotations = mutable.HashMap[Named,Annotation]()
    def handleInlineOption(value: String): Unit =
       value.split('.') match {
         case Array(module) =>
           inlineAnnotations(ModuleName(module)) = TagAnnotation
         case Array(module, inst) =>
           inlineAnnotations(ComponentName(inst,ModuleName(module))) = TagAnnotation
         case _ => throw new Exception(s"Bad inline instance/module name: $value")
       }

    type OptionMap = Map[CompilerOption, String]
    def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      list match {
        case Nil => map
        case "--inline" :: value :: tail =>
          handleInlineOption(value)
          nextOption(map, tail)
        case "-X" :: value :: tail =>
                  nextOption(map + (CompilerName -> value), tail)
        case "-i" :: value :: tail =>
                  nextOption(map + (InputFileName -> value), tail)
        case "-o" :: value :: tail =>
                  nextOption(map + (OutputFileName -> value), tail)
        case "--info-mode" :: value :: tail =>
                  nextOption(map + (InfoModeOption -> value), tail)
        case ("-h" | "--help") :: tail => { println(usage); sys.exit(0) }
        case option :: tail =>
                  throw new Exception("Unknown option " + option)
      }
    }

    val options = nextOption(defaultOptions, arglist)

    // Get input circuit/output filenames
    val input = options.getOrElse(InputFileName, throw new Exception("No input file provided!" + usage))
    val output = options.getOrElse(OutputFileName, throw new Exception("No output file provided!" + usage))

    val infoMode = options.get(InfoModeOption) match {
      case (Some("use") | None) => UseInfo
      case Some("ignore") => IgnoreInfo
      case Some("gen") => GenInfo(input)
      case Some("append") => AppendInfo(input)
      case Some(other) => throw new Exception("Unknown info mode option: " + other)
    }

    // Construct all Circuit Annotations
    val inlineCA =
       if (inlineAnnotations.isEmpty) Seq.empty
       else Seq(StickyCircuitAnnotation(passes.InlineCAKind, inlineAnnotations.toMap))
    val allAnnotations = inlineCA // other annotations will be added here

    // Execute selected compiler - error if not recognized compiler
    options.get(CompilerName) match {
      case Some("high") => compile(input, output, new HighFirrtlCompiler(), infoMode, allAnnotations)
      case Some("low") => compile(input, output, new LowFirrtlCompiler(), infoMode, allAnnotations)
      case Some("verilog") => compile(input, output, new VerilogCompiler(), infoMode, allAnnotations)
      case Some(other) => throw new Exception("Unknown compiler option: " + other)
      case None => throw new Exception("No specified compiler option.")
    }
  }
}
