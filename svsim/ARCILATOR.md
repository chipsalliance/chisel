# Arcilator backend

## Running the repository tests

Set `CHISELSIM_SIMULATOR` once for the entire test process:

```sh
CHISELSIM_SIMULATOR=arcilator ./mill 'chisel[2.13].test' -oF
CHISELSIM_SIMULATOR=verilator ./mill 'chisel[2.13].test' -oF
```

Use `chisel[3].test` for Scala 3, or `test.testOnly` to select individual
suites. The setting applies to ChiselSim, EphemeralSimulator, the ChiselTest
compatibility API, and the low-level simulator/DPI tests. Verilator remains
the default when the variable is unset. An explicit `HasSimulator`, backend
annotation, or ScalaTest `-Dsimulator` takes precedence. Invalid environment
values fail instead of silently selecting another simulator.

The reusable `.github/workflows/test.yml` accepts `simulator: arcilator` or
`simulator: verilator` (the default). Manual runs of `ci.yml` expose the same
choice. Once backend parity is achieved, a caller can select by event:

```yaml
with:
  system: ubuntu-24.04
  jvm: 21
  scala: '2.13'
  simulator: ${{ github.event_name == 'pull_request' && 'arcilator' || 'verilator' }}
```

Selecting Arcilator installs the full CIRCT bundle. The current backend does
**not yet pass the entire repository suite**: async resets, DPI, plusargs,
delayed logic, and some verification-layer lowering are unsupported. These
tests expose failures rather than falling back to Verilator. The FST-specific
test is canceled under Arcilator. The shared svsim backend tests follow the
same environment switch; Verilator coverage and FST tests still exercise
Verilator. Keep the required CI job on Verilator
until these backend limitations are resolved.

## Selecting a simulator in code

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

The backend finds the executable via `ARCILATOR_PATH`, then `PATH`, then
`firtool-1.163.0/bin/arcilator` relative to the project root
(`svsim.Workspace.getProjectRootOrCwd`). `ARCILATOR_PATH` is the path to the
executable, not the bundle directory. It uses `circt-verilog`, `circt-opt`,
`llc`, `llvm-config`, the Arc runtime archive, and headers from the same bundle.
A C++17 compiler (`CXX`, default `clang++`) is required. Compilation and native
bridge generation are implemented in Scala; the replay Makefile invokes the
Scala compiler entry point using the current JVM and classpath.

Compilation imports the existing SystemVerilog sources into CIRCT, compiles
them to LLVM, and generates a native bridge for svsim's command protocol.
Compilation artifacts and the replay Makefile are in `workdir-arcilator`.
VCD tracing supports ports and observable registers and uses the configured
trace file stem and timescale. Other waveform formats and Verilog plusargs are
rejected. Verilog features are limited by the bundled CIRCT frontend and
arcilator; in particular, asynchronous resets are not supported by this release.
Delayed FIRRTL random initialization is replaced by native register
and memory initialization before simulation begins.
