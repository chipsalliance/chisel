// SPDX-License-Identifier: Apache-2.0

package svsim.arcilator

import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path, Paths}
import scala.collection.mutable.ArrayBuffer
import scala.sys.process._

/** The Makefile entry point for compiling a native arcilator simulation.
  *
  * Keeping the orchestration in Scala lets both initial compilation and `make
  * replay` use the same code, without requiring an additional scripting runtime.
  */
object Compiler {
  private def read(path:  Path):                   String = new String(Files.readAllBytes(path), UTF_8)
  private def write(path: Path, contents: String): Unit = { Files.write(path, contents.getBytes(UTF_8)); () }

  private def run(arguments: Seq[String]): Unit = {
    val status =
      Process(arguments).!(ProcessLogger(line => Console.out.println(line), line => Console.err.println(line)))
    require(status == 0, s"Command failed with exit code $status: ${arguments.mkString(" ")}")
  }

  def main(arguments: Array[String]): Unit = {
    val options = ArrayBuffer.empty[(String, String)]
    var index = 0
    while (index < arguments.length) {
      val argument = arguments(index)
      if (argument == "--vcd") options += argument -> "true"
      else if (argument.contains("=")) {
        val parts = argument.split("=", 2)
        options += parts(0) -> parts(1)
      } else {
        require(index + 1 < arguments.length, s"Missing value for $argument")
        index += 1
        options += argument -> arguments(index)
      }
      index += 1
    }
    def all(name:      String): Seq[String] = options.collect { case (`name`, value) => value }.toSeq
    def required(name: String): String = all(name).headOption.getOrElse(sys.error(s"Missing $name"))

    val tools = Paths.get(required("--tools"))
    val generated = Paths.get("../generated-sources")
    val manifest = read(generated.resolve("arcilator-ports.txt")).linesIterator.toSeq
    val top = manifest.head
    val ports = manifest.tail.map { line =>
      val fields = line.split("\t")
      svsim.ModuleInfo.Port(fields(0), isSettable = fields(1).toBoolean, isGettable = fields(2).toBoolean)
    }
    val frontendOptions = all("--frontend-option").map(_.replace("svsimTestbench.", top + "."))
    val randomizeRegisters = frontendOptions.contains("-DRANDOMIZE_REG_INIT")
    val randomizeMemories = frontendOptions.contains("-DRANDOMIZE_MEM_INIT")
    // Initialize native storage instead of importing firtool's delayed initial blocks.
    val frontend = frontendOptions.filterNot { option =>
      option.startsWith("-DRANDOMIZE") || option.startsWith("-DRANDOM=") ||
      (option.startsWith("-Dlayer$") && option.contains("$Temporal"))
    }
    val sources = read(Paths.get(required("-F"))).linesIterator.filter { line =>
      (line.endsWith(".sv") || line.endsWith(".v")) &&
      Paths.get(line).toAbsolutePath.normalize != generated.resolve("testbench.sv").toAbsolutePath.normalize
    }.toSeq
    run(
      Seq(tools.resolve("circt-verilog").toString, "--ir-hw", s"--top=$top", "--single-unit") ++
        frontend ++ sources ++ Seq("-o", "model.mlir")
    )
    // Hoist hierarchical probes, including extracted Chisel verification layers,
    // before arcilator converts LLHD processes to arcs.
    run(
      Seq(
        tools.resolve("circt-opt").toString,
        "model.mlir",
        "--hw-flatten-modules=hw-inline-public",
        "--llhd-sig2reg",
        "--llhd-hoist-signals",
        "--canonicalize",
        "-o",
        "model.mlir"
      )
    )
    lowerTermination(Paths.get("model.mlir"))
    val enableVcd = all("--vcd").nonEmpty
    run(
      Seq(
        tools.resolve("arcilator").toString,
        "model.mlir",
        "--emit-llvm",
        "--observe-ports",
        "--no-generate-driver",
        "--state-file=model.json"
      ) ++
        Option.when(enableVcd || randomizeRegisters)("--observe-registers") ++
        Option.when(randomizeMemories)("--observe-memories") ++ all("--arc-option") ++ Seq("-o", "model.ll")
    )
    val model = ujson
      .read(read(Paths.get("model.json")))
      .arr
      .find(_("name").str == top)
      .getOrElse(sys.error(s"Arcilator did not emit model $top"))
    val timescale = frontend.find(_.startsWith("--timescale=")).map(_.split("/", 2)(1)).getOrElse("1ns")
    write(
      Paths.get("arcilator-bridge.cpp"),
      Bridge.generate(model, ports, enableVcd, timescale, randomizeRegisters, randomizeMemories)
    )
    // System clang may not understand the LLVM IR emitted by this arcilator release.
    val optimization = all("--optimization").headOption.getOrElse("-O3")
    run(
      Seq(
        tools.resolve("llc").toString,
        "model.ll",
        "-filetype=obj",
        "-relocation-model=pic",
        optimization,
        "-o",
        "model.o"
      )
    )
    val llvmFlags = Process(
      Seq(tools.resolve("llvm-config").toString, "--link-static", "--libs", "support", "--system-libs")
    ).!!.trim.split("\\s+").filter(_.nonEmpty).toSeq
    run(
      Seq(
        sys.env.getOrElse("CXX", "clang++"),
        "-std=c++17",
        optimization,
        "-DSVSIM_ENABLE_ARCILATOR_SUPPORT",
        "-I" + tools.getParent.resolve("include"),
        generated.resolve("simulation-driver.cpp").toString,
        "arcilator-bridge.cpp",
        "model.o",
        tools.getParent.resolve("lib/libCIRCTArcRuntime.a").toString,
        "-L" + tools.getParent.resolve("lib")
      ) ++
        llvmFlags ++ all("--link-library") ++ Seq("-o", required("--output"))
    )
  }

  private def lowerTermination(path: Path): Unit = {
    // This release cannot lower sim.terminate in an LLHD process. Use a native
    // callback to preserve successful finish and fatal failure semantics.
    val termination = "(?m)^([ \\t]*)sim\\.terminate (failure|success)(?:, quiet)?[ \\t]*$".r
    val ir = read(path)
    val lowered = termination.replaceAllIn(
      ir,
      m => {
        val status = if (m.group(2) == "failure") 1 else 0
        s"${m.group(1)}%svsim_exit_${m.start} = arith.constant $status : i32\n" +
          s"${m.group(1)}func.call @svsim_arc_exit(%svsim_exit_${m.start}) : (i32) -> ()"
      }
    )
    write(
      path,
      if (lowered == ir) ir
      else
        lowered.replaceFirst("module \\{", "module {\n  func.func private @svsim_arc_exit(i32)")
    )
  }
}
