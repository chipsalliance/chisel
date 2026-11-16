#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compile an arcilator model and generate its svsim native port bridge."""

import argparse
import json
import os
import re
import shlex
import sys
from pathlib import Path
import subprocess


def run(*args):
    subprocess.run([str(arg) for arg in args], check=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tools", type=Path, required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--optimization", default="-O3")
parser.add_argument("--vcd", action="store_true")
parser.add_argument("--frontend-option", action="append", default=[])
parser.add_argument("--arc-option", action="append", default=[])
parser.add_argument("--link-library", action="append", default=[])
parser.add_argument("-F", dest="filelist", type=Path, required=True)
args = parser.parse_args()
generated = Path("../generated-sources")
manifest = (generated / "arcilator-ports.txt").read_text().splitlines()
top = manifest[0]
ports = [line.split("\t") for line in manifest[1:]]
args.frontend_option = [option.replace("svsimTestbench.", top + ".")
                        for option in args.frontend_option]
randomize_registers = "-DRANDOMIZE_REG_INIT" in args.frontend_option
randomize_memories = "-DRANDOMIZE_MEM_INIT" in args.frontend_option
# Randomize native storage instead of importing firtool's delayed initial blocks.
args.frontend_option = [option for option in args.frontend_option
                        if not option.startswith(("-DRANDOMIZE", "-DRANDOM="))
                        and not (option.startswith("-Dlayer$") and "$Temporal" in option)]
sources = [line for line in args.filelist.read_text().splitlines()
           if Path(line).suffix in (".sv", ".v")
           and Path(line).resolve() != (generated / "testbench.sv").resolve()]
run(args.tools / "circt-verilog", "--ir-hw", "--top=" + top,
    "--single-unit", *args.frontend_option, *sources, "-o", "model.mlir")
# Hoist hierarchical probes (including extracted Chisel verification layers)
# before arcilator converts LLHD processes to arcs.
run(args.tools / "circt-opt", "model.mlir", "--hw-flatten-modules=hw-inline-public",
    "--llhd-sig2reg", "--llhd-hoist-signals", "--canonicalize", "-o", "model.mlir")
# This arcilator release cannot lower sim.terminate in an LLHD process. Lower
# it to a native callback, preserving both successful finish and fatal failure.
ir = Path("model.mlir").read_text()
def lower_termination(match):
    status = 1 if match.group(2) == "failure" else 0
    return (f'{match.group(1)}%svsim_exit_{match.start()} = arith.constant {status} : i32\n'
            f'{match.group(1)}func.call @svsim_arc_exit(%svsim_exit_{match.start()}) : (i32) -> ()')
ir, termination_count = re.subn(r'(?m)^(\s*)sim\.terminate (failure|success)(?:, quiet)?\s*$',
                               lower_termination, ir)
if termination_count:
    ir = ir.replace("module {", "module {\n  func.func private @svsim_arc_exit(i32)", 1)
Path("model.mlir").write_text(ir)
run(args.tools / "arcilator", "model.mlir", "--emit-llvm", "--observe-ports",
    "--no-generate-driver", "--state-file=model.json",
    *(["--observe-registers"] if args.vcd or randomize_registers else []),
    *(["--observe-memories"] if randomize_memories else []),
    *args.arc_option, "-o", "model.ll")
models = json.loads(Path("model.json").read_text())
model = next(m for m in models if m["name"] == top)
states = {(s["name"], s["type"]): s for s in model["states"]}

code = [r'''
#include <cstdint>
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
''']

# Use asm labels because Verilog module names need not be C++ identifiers.
symbol_prefix = "_" if sys.platform == "darwin" else ""
code += [f'extern "C" void model_eval(uint8_t *) asm({json.dumps(symbol_prefix + top + "_eval")});',
         f'extern "C" ArcRuntimeModelInfo model_info asm({json.dumps(symbol_prefix + "arcRuntimeModel_" + top)});']
for kind in ("initial", "final"):
    symbol = model.get(kind + "FnSym", "")
    if symbol:
        code.append(f'extern "C" void model_{kind}(uint8_t *) asm({json.dumps(symbol_prefix + symbol)});')

for index, (name, settable, gettable) in enumerate(ports):
    state = states.get((name, "input" if settable == "true" else "output"))
    if state is None:
        raise RuntimeError(f"Arcilator did not expose port {name}")
    width = state["numBits"]
    size = (width + 7) // 8
    offset = state["offset"]
    if gettable == "true":
        code.append(f'''static void get_{index}(uint8_t *value) {{
  memset(value, 0, {(width + 31) // 32 * 4});
  memcpy(value, modelState + {offset}, {size});
  value[{size - 1}] &= {255 >> ((8 - width % 8) % 8)};
}}''')
    if settable == "true":
        code.append(f'''static void set_{index}(const uint8_t *value) {{
  memcpy(modelState + {offset}, value, {size});
  modelState[{offset + size - 1}] &= {255 >> ((8 - width % 8) % 8)};
}}''')

for kind, direction, signature in (
    ("getter", 2, "uint8_t *"), ("setter", 1, "const uint8_t *")
):
    code.append(f'extern "C" int port_{kind}(int id, int *width, void (**fn)({signature})) {{\nswitch (id) {{')
    for index, port in enumerate(ports):
        if port[direction] == "true":
            state = states[(port[0], "input" if port[1] == "true" else "output")]
            operation = "get" if kind == "getter" else "set"
            code.append(f'case {index}: *width = {state["numBits"]}; *fn = {operation}_{index}; return 0;')
    code.append('default: return -1;\n}\n}')

trace_states = [s for s in model["states"] if s["type"] in ("input", "output", "register")]
code.append('static TraceSignal traceSignals[] = {')
for state in trace_states:
    code.append(f'{{{json.dumps(state["name"])}, {state["offset"]}, {state["numBits"]}}},')
# A sentinel allows even a portless design to have a legal C++ array.
code.append('{nullptr, 0, 0}};')
code.append(r'''
static void dumpTrace() {
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
''')
if args.vcd:
    timescale = next((s.split("=", 1)[1].split("/")[1]
                      for s in args.frontend_option if s.startswith("--timescale=")), "1ns")
    code.append(f'''
  trace = fopen((std::string(path) + ".vcd").c_str(), "w");
  if (!trace) {{ perror("Cannot open VCD trace"); exit(1); }}
  fprintf(trace, "$timescale {timescale} $end\\n$scope module dut $end\\n");
  for (unsigned i = 0; traceSignals[i].name; ++i)
    fprintf(trace, "$var wire %u p%u %s $end\\n", traceSignals[i].width, i, traceSignals[i].name);
  fprintf(trace, "$upscope $end\\n$enddefinitions $end\\n");
''')
code.append(r'''
}
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
''')
for state in model["states"]:
    if ((state["type"] == "register" and randomize_registers) or
            (state["type"] == "memory" and randomize_memories)):
        size = (state["numBits"] + 7) // 8
        depth = state.get("depth", 1)
        stride = state.get("stride", size)
        code.append(f'''for (unsigned word = 0; word < {depth}; ++word)
  for (unsigned byte = 0; byte < {size}; ++byte)
    modelState[{state["offset"]} + word * {stride} + byte] = rand();''')
if model.get("initialFnSym"):
    code.append('model_initial(modelState);')
code.append('arcRuntimeOnInitialized(instance);\nsimulation_body();\nsimulation_final();')
if model.get("finalFnSym"):
    code.append('model_final(modelState);')
code.append('if (trace) fclose(trace);\narcRuntimeDeleteInstance(instance);\n}')
Path("arcilator-bridge.cpp").write_text("\n".join(code))

# Use the bundled LLVM compiler: system clang may not understand the LLVM IR
# emitted by this version of arcilator.
run(args.tools / "llc", "model.ll", "-filetype=obj", "-relocation-model=pic",
    "-O0" if args.optimization == "-O0" else "-O3", "-o", "model.o")
llvm_link_flags = shlex.split(subprocess.check_output(
    [str(args.tools / "llvm-config"), "--link-static", "--libs", "support", "--system-libs"],
    text=True))
run(os.environ.get("CXX", "clang++"), "-std=c++17", args.optimization,
    "-DSVSIM_ENABLE_ARCILATOR_SUPPORT", "-I" + str(args.tools.parent / "include"),
    generated / "simulation-driver.cpp", "arcilator-bridge.cpp", "model.o",
    args.tools.parent / "lib/libCIRCTArcRuntime.a", "-L" + str(args.tools.parent / "lib"),
    *llvm_link_flags, *args.link_library,
    "-o", args.output)
