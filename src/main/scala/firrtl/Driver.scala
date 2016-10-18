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

import scala.io.Source
import scala.collection.mutable
import Annotations._

import Utils._
import Parser.{InfoMode, IgnoreInfo, UseInfo, GenInfo, AppendInfo}

object Driver {
  /**
   * Implements the default Firrtl compilers and an inlining pass.
   *
   * Arguments specify the compiler, input file, output file, and
   * optionally the module/instances to inline.
   */
  def main(args: Array[String]) = {
    val usage = """
Usage: firrtl -i <input_file> -o <output_file> -X <compiler> [options] 
       sbt "run-main firrtl.Driver -i <input_file> -o <output_file> -X <compiler> [options]"

Required Arguments:
  -i <filename>         Specify the input *.fir file
  -o <filename>         Specify the output file
  -X <compiler>         Specify the target compiler
                        Currently supported: high low verilog

Optional Arguments:
  --info-mode <mode>             Specify Info Mode
                                 Supported modes: ignore, use, gen, append
  --inferRW <circuit>            Enable readwrite port inference for the target circuit
  --inline <module>|<instance>   Inline a module (e.g. "MyModule") or instance (e.g. "MyModule.myinstance")

  --replSeqMem -c:<circuit>:-i:<filename>:-o:<filename> 
  *** Replace sequential memories with blackboxes + configuration file
  *** Input configuration file optional
  *** Note: sub-arguments to --replSeqMem should be delimited by : and not white space!
  
  [--help|-h]                    Print usage string
"""

    def handleInlineOption(value: String): Annotation =
      value.split('.') match {
        case Array(circuit) =>
          passes.InlineAnnotation(CircuitName(circuit), TransID(0))
        case Array(circuit, module) =>
          passes.InlineAnnotation(ModuleName(module, CircuitName(circuit)), TransID(0))
        case Array(circuit, module, inst) =>
          passes.InlineAnnotation(ComponentName(inst, ModuleName(module, CircuitName(circuit))), TransID(0))
        case _ => throw new Exception(s"Bad inline instance/module name: $value" + usage)
      }

    def handleInferRWOption(value: String) = 
      passes.InferReadWriteAnnotation(value, TransID(-1))

    def handleReplSeqMem(value: String) = 
      passes.memlib.ReplSeqMemAnnotation(value, TransID(-2))

    run(args: Array[String],
      Map( "high" -> new HighFirrtlCompiler(),
        "low" -> new LowFirrtlCompiler(),
        "verilog" -> new VerilogCompiler()),
      Map("--inline" -> handleInlineOption _,
          "--inferRW" -> handleInferRWOption _,
          "--replSeqMem" -> handleReplSeqMem _),
      usage
    )
  }


  // Compiles circuit. First parses a circuit from an input file,
  //  executes all compiler passes, and writes result to an output
  //  file.
  def compile(
      input: String, 
      output: String, 
      compiler: Compiler, 
      infoMode: InfoMode = IgnoreInfo,
      annotations: AnnotationMap = new AnnotationMap(Seq.empty)) = {
    val parsedInput = Parser.parse(Source.fromFile(input).getLines, infoMode)
    val outputBuffer = new java.io.CharArrayWriter
    compiler.compile(parsedInput, annotations, outputBuffer)

    val outputFile = new java.io.PrintWriter(output)
    outputFile.write(outputBuffer.toString)
    outputFile.close()
  }

  /**
   * Runs a Firrtl compiler.
   *
   * @param args list of commandline arguments
   * @param compilers mapping a compiler name to a compiler
   * @param customOptions mapping a custom option name to a function that returns an annotation
   * @param usage describes the commandline API
   */
  def run(args: Array[String], compilers: Map[String,Compiler], customOptions: Map[String, String=>Annotation], usage: String) = {
    /**
     * Keys commandline values specified by user in OptionMap
     */
    sealed trait CompilerOption
    case object InputFileName extends CompilerOption
    case object OutputFileName extends CompilerOption
    case object CompilerName extends CompilerOption
    case object AnnotationOption extends CompilerOption
    case object InfoModeOption extends CompilerOption
    /**
     * Maps compiler option to user-specified value
     */
    type OptionMap = Map[CompilerOption, String]

    /**
     * Populated by custom annotations returned from corresponding function
     * held in customOptions
     */
    val annotations = mutable.ArrayBuffer[Annotation]()
    def nextOption(map: OptionMap, list: List[String]): OptionMap = {
      list match {
        case Nil => map
        case "-i" :: value :: tail =>
          nextOption(map + (InputFileName -> value), tail)
        case "-o" :: value :: tail =>
          nextOption(map + (OutputFileName -> value), tail)
        case "-X" :: value :: tail =>
          nextOption(map + (CompilerName -> value), tail)
        case "--info-mode" :: value :: tail =>
          nextOption(map + (InfoModeOption -> value), tail)
        case flag :: value :: tail if customOptions.contains(flag) =>
          annotations += customOptions(flag)(value)
          nextOption(map, tail)
        case ("-h" | "--help") :: tail => println(usage); sys.exit(0)
        case option :: tail =>
          throw new Exception("Unknown option " + option + usage)
      }
    }

    val arglist = args.toList
    val options = nextOption(Map[CompilerOption, String](), arglist)

    // Get input circuit/output filenames
    val input = options.getOrElse(InputFileName, throw new Exception("No input file provided!" + usage))
    val output = options.getOrElse(OutputFileName, throw new Exception("No output file provided!" + usage))

    val infoMode = options.get(InfoModeOption) match {
      case (Some("append") | None) => AppendInfo(input)
      case Some("use") => UseInfo
      case Some("ignore") => IgnoreInfo
      case Some("gen") => GenInfo(input)
      case Some(other) => throw new Exception("Unknown info mode option: " + other + usage)
    }

    // Execute selected compiler - error if not recognized compiler
    options.get(CompilerName) match {
      case Some(name) =>
        compilers.get(name) match {
          case Some(compiler) => compile(input, output, compiler, infoMode, new AnnotationMap(annotations.toSeq))
          case None => throw new Exception("Unknown compiler option: " + name + usage)
        }
      case None => throw new Exception("No specified compiler option." + usage)
    }
  }
}
