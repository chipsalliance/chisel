// SPDX-License-Identifier: Apache-2.0

package svsim.arcilator

import svsim.ModuleInfo

/** Generate the native port accessors and simulation hooks from arcilator's state layout. */
private[arcilator] object Bridge {
  def generate(
    model:              ujson.Value,
    ports:              Seq[ModuleInfo.Port],
    enableVcd:          Boolean,
    timescale:          String,
    randomizeRegisters: Boolean,
    randomizeMemories:  Boolean
  ): String = {
    val code = new StringBuilder
    def emit(text:    String):                     Unit = { code.append(text).append('\n'); () }
    def quote(text:   String):                     String = ujson.write(text)
    def number(state: ujson.Value, field: String): Int = state(field).num.toInt
    val states = model("states").arr.toSeq
    val portStates = states.map(s => (s("name").str, s("type").str) -> s).toMap
    def portState(port: ModuleInfo.Port): ujson.Value =
      portStates.getOrElse(
        (port.name, if (port.isSettable) "input" else "output"),
        sys.error(s"Arcilator did not expose port ${port.name}")
      )
    def function(kind: String): Option[String] = model.obj.get(kind + "FnSym").map(_.str).filter(_.nonEmpty)

    emit("""#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <csetjmp>
#include "circt/Dialect/Arc/Runtime/ArcRuntime.h"
extern "C" void simulation_body();
extern "C" void simulation_final();
static ArcState *instance;
static uint8_t *modelState;
static uint64_t simulationTime;
static FILE *trace;
static bool traceEnabled;
static bool finished;
static std::jmp_buf finishJump;
extern "C" void svsim_arc_exit(int status) {
  fflush(nullptr);
  if (status) exit(status);
  finished = true;
  std::longjmp(finishJump, 1);
}
struct TraceSignal { const char *name; unsigned offset; unsigned width; };
""")
    // asm labels accommodate module names that are not C++ identifiers.
    val prefix = if (sys.props("os.name").toLowerCase.contains("mac")) "_" else ""
    val top = model("name").str
    emit(s"""extern "C" void model_eval(uint8_t *) asm(${quote(prefix + top + "_eval")});""")
    emit(s"""extern "C" ArcRuntimeModelInfo model_info asm(${quote(prefix + "arcRuntimeModel_" + top)});""")
    for {
      kind <- Seq("initial", "final")
      symbol <- function(kind)
    } {
      emit(s"""extern "C" void model_$kind(uint8_t *) asm(${quote(prefix + symbol)});""")
    }
    for ((port, index) <- ports.zipWithIndex) {
      val state = portState(port)
      val width = number(state, "numBits")
      val size = (width + 7) / 8
      val offset = number(state, "offset")
      val mask = 255 >> ((8 - width % 8) % 8)
      if (port.isGettable) emit(s"""static void get_$index(uint8_t *value) {
  memset(value, 0, ${(width + 31) / 32 * 4});
  memcpy(value, modelState + $offset, $size);
  value[${size - 1}] &= $mask;
}""")
      if (port.isSettable) emit(s"""static void set_$index(const uint8_t *value) {
  memcpy(modelState + $offset, value, $size);
  modelState[${offset + size - 1}] &= $mask;
}""")
    }
    for (getter <- Seq(true, false)) {
      val kind = if (getter) "getter" else "setter"
      val signature = if (getter) "uint8_t *" else "const uint8_t *"
      val operation = if (getter) "get" else "set"
      emit(s"""extern "C" int port_$kind(int id, int *width, void (**fn)($signature)) {
switch (id) {""")
      for ((port, index) <- ports.zipWithIndex if (if (getter) port.isGettable else port.isSettable)) {
        emit(s"case $index: *width = ${number(portState(port), "numBits")}; *fn = ${operation}_$index; return 0;")
      }
      emit("default: return -1;\n}\n}")
    }

    emit("static TraceSignal traceSignals[] = {")
    for (state <- states if Set("input", "output", "register")(state("type").str)) {
      emit(s"{${quote(state("name").str)}, ${number(state, "offset")}, ${number(state, "numBits")}},")
    }
    // Sentinel makes this a legal array even for a portless design.
    emit("{nullptr, 0, 0}};")
    emit("""static void dumpTrace() {
  if (!trace || !traceEnabled) return;
  fprintf(trace, "#%llu\n", (unsigned long long)simulationTime);
  for (unsigned i = 0; traceSignals[i].name; ++i) {
    auto &signal = traceSignals[i];
    fputc('b', trace);
    for (unsigned bit = signal.width; bit > 0; --bit)
      fputc((modelState[signal.offset + (bit - 1) / 8] >> ((bit - 1) % 8)) & 1 ? '1' : '0', trace);
    fprintf(trace, " p%u\n", i);
  }
  fflush(trace);
}
extern "C" void simulation_initializeTrace(const char *path) {
""")
    if (enableVcd) {
      emit("""  trace = fopen((std::string(path) + ".vcd").c_str(), "w");
  if (!trace) { perror("Cannot open VCD trace"); exit(1); }""")
      emit(s"  fprintf(trace, ${quote(s"$$timescale $timescale $$end\n$$scope module dut $$end\n")});")
      emit("""  for (unsigned i = 0; traceSignals[i].name; ++i)
    fprintf(trace, "$var wire %u p%u %s $end\n", traceSignals[i].width, i, traceSignals[i].name);
  fprintf(trace, "$upscope $end\n$enddefinitions $end\n");""")
    }
    emit("""}
extern "C" void simulation_enableTrace(int *success) {
  *success = trace != nullptr;
  traceEnabled = true;
  dumpTrace();
}
extern "C" void simulation_disableTrace(int *success) {
  *success = trace != nullptr;
  traceEnabled = false;
}
extern "C" void run_simulation(int delay, int *finish) {
  if (!finished && setjmp(finishJump) == 0) {
    memcpy(modelState, &simulationTime, sizeof(simulationTime));
    arcRuntimeOnEval(instance);
    model_eval(modelState);
  }
  dumpTrace();
  simulationTime += delay;
  *finish = finished;
}
extern "C" void simulation_main(int argc, const char **argv) {
  instance = arcRuntimeAllocateInstance(&model_info, nullptr);
  modelState = instance->modelState;
""")
    for (
      state <- states if (state("type").str == "register" && randomizeRegisters) ||
        (state("type").str == "memory" && randomizeMemories)
    ) {
      val size = (number(state, "numBits") + 7) / 8
      val depth = state.obj.get("depth").map(_.num.toInt).getOrElse(1)
      val stride = state.obj.get("stride").map(_.num.toInt).getOrElse(size)
      emit(s"""for (unsigned word = 0; word < $depth; ++word)
  for (unsigned byte = 0; byte < $size; ++byte)
    modelState[${number(state, "offset")} + word * $stride + byte] = rand();""")
    }
    if (function("initial").nonEmpty) emit("model_initial(modelState);")
    emit("arcRuntimeOnInitialized(instance);\nsimulation_body();\nsimulation_final();")
    if (function("final").nonEmpty) emit("model_final(modelState);")
    emit("if (trace) fclose(trace);\narcRuntimeDeleteInstance(instance);\n}")
    code.toString
  }
}
