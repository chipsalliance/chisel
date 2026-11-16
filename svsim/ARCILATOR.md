# Arcilator backend

ChiselSim can select arcilator with:

```scala
implicit val simulator: chisel3.simulator.HasSimulator =
  chisel3.simulator.HasSimulator.simulators.arcilator()
```

Tests mixing in `chisel3.simulator.scalatest.Cli.Simulator` also accept
`-Dsimulator=arcilator`, with `-DemitVcd=1` to enable waveform dumping.

The ChiselTest compatibility API supports:

```scala
test(new MyModule)
  .withAnnotations(Seq(ArcilatorBackendAnnotation, WriteVcdAnnotation)) { dut =>
    dut.clock.step()
  }
```

Alternatively, override `ChiselScalatestTester.simulator` to select arcilator for
all tests in a suite.

The backend currently uses `firtool-1.163.0/bin/arcilator` relative to the project
root (`svsim.Workspace.getProjectRootOrCwd`). It uses `circt-verilog`, `circt-opt`,
`llc`, `llvm-config`, the Arc runtime archive, and headers from the same bundle.
Python 3 and a C++17 compiler (`CXX`, default `clang++`) are required.

Compilation imports the existing SystemVerilog sources into CIRCT, compiles
them to LLVM, and generates a native bridge for svsim's command protocol.
Compilation artifacts and the replay Makefile are in `workdir-arcilator`.
VCD tracing supports ports and observable registers and uses the configured
trace file stem and timescale. Other waveform formats and Verilog plusargs are
rejected. Verilog features are limited by the bundled CIRCT frontend and
arcilator; in particular, asynchronous resets are not supported by this release.
Delayed FIRRTL random initialization is replaced by native register
and memory initialization before simulation begins.
