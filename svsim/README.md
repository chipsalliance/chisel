# svsim

`svsim` is a low-level library for compiling and controlling SystemVerilog
simulations, currently targeting Verilator and VCS as backends.

## Design

In no particular order, here are some of the design considerations of `svsim`:
* `svsim` does not know about Chisel. The inputs to `svsim` are SystemVerilog
  files and the `ModuleInfo` class, which encodes all the information `svsim`
  needs to generate a test harness for the supplied `SystemVerilog`.
* `svsim` **is not a testing framework**, and is not opinionated about how a
  higher-level testing framework should be built. `svsim` should enable
  SystemVerilog simulation for any testing framework and should not be tied to
  any particular technology (i.e. Scalatest).
* `svsim` provides a backend-independent test harness. This means that `svsim`'s
  generated sources are not impacted by the choice of backend and indeed can be
  shared between multiple backends. If backend-specific behavior is required, it
  is provided via a compile-time flag (such as
  `SVSIM_BACKEND_SUPPORTS_DELAY_IN_PUBLIC_FUNCTIONS`).
* `svsim` simulations are also backend-independent, meaning that the same code
  launches and controls a simulation regardless of what backend it was compiled
  using. 
* `svsim` backends are declarative. This means a backend only controls which
  executable is invoked for compilation, and what arguments are provided to that
  executable (either via command line or environment variables). Backends may
  also provide arguments to the simulation, but may not have any other
  specialized logic.
* Simulations are controlled using an ad-hoc protocol where commands are read
  from `stdin` and messages are written to `stdout`. This protocol is **not**
  meant to be an ABI, and can change freely between Chisel verisons.
* Communication with the simulation is pipelined, so the Scala driver can send
  multiple commands before processing responses from the simulation. This allows
  `svsim` to effectively eliminate the overhead of communicating with the
  simulation without resorting to JNA or Scala-side caching. This optimization
  can, of course, be defeated if the driver code waits on every command, but we
  consider this an antipattern.
* `svsim` strives to have the minimum amount of dependencies possible, so that
  it may be maximally portable.
* `svsim` does not provide syntactic sugar, and expects that higher level
  frameworks will implement whatever sugar they thing is appropriate. For
  example, if a framework like `chiseltest` were to use `svsim` under the hood,
  it might choose to store the `Simulation.Controller` in a `DynamicValue` to
  enable calling `peek` and `poke` directly on a Chisel `Data`, or pass
  information via annotations. Such specialization should stay in the
  higher-level framework and all `svsim` APIs should stay explicit.

## Architecture

We consider it a top priority to keep `svsim`'s architecture and implementation
as simple as possible. In support of this effort, there are really only three
components to `svsim`:

### Workspace

The `Workspace` manages all interaction with the filesystem. Its API is mutable,
because the underlying representation (files on disk) is mutable. A `Workspace`
is represented on the filesystem as a folder. The `reset()` method will delete
any previous state and create the necessary folders. The `elaborate()` method
takes a `ModuleInfo` which describes the SystemVerilog module to be simulated.
The idea is that a higher-level framework will provide specialized `elaborate`
methods which emit SystemVerilog to `primarySourcesPath`, similar to what we do
with `elaborateGCD()` in `src/test/scala/Resources.scala`. For example, Chisel
can provide an `elaborate[T <: RawModule](module: => T)` method.
`generateAdditionalSources()` uses the `ModuleInfo` from `elaborate` and
generates the test harness. Finally, `compile()` uses a `Backend` to compile the
simulation. Because `svsim`'s test harness is backend-independent, `compile` may
be called multiple times with different settings or different backends to create
multiple `Simulation`s (a `workingDirectoryTag` is provided so that mutliple
simulations are output to separate directories). 

### Simulation

The `Workspace`'s `compile` method returns a `Simulation` instance. A
`Simulation` is launched using the `run` method, which passes a
`Simulation.Controller` instance to the provided closure. The
`Simulation.Controller` is used to explicitly control the `Simulation`, and
should not escape the body of the closure. Like most of `svsim`,
`Simulation.Controller` aims to be a low level API on which higher level APIs
can be built. For example, `Simulation.Controller`'s `get` method does not
implicitly run the simulation to make the effects of previous `set` calls
visible, the way you would expect for a peek/poke test. To evaluate the
simulation state, the user must explicitly call `controller.run(0)`. This, of
course, does not prevent a peek/poke style API from being built using these
building blocks in a higher-level framework.

### Backend

`svsim` currently provides two `Backend`s: `verilator.Backend` and
`vcs.Backend`. Settings which **all** `svsim` `Backend`s must support (like
SystemVerilog preprocessor defines) live in `CommonCompilationSettings`, and each
backend has its own backend-specific `CompilationSettings`.

## Advanced Usage

### `make simulation`

Because `Backend`s are declarative, `svsim` is able to output a `Makefile` for
compiling the simulation. This `Makefile` can be re-invoked after the fact to
rebuild the simulation without re-running the Scala driver (`make simulation`).
Users are free to modify either the harness generated by `svsim`
(`generated-sources`), the SystemVerilog provided to `svsim`
(`primary-sources`), or even the arguments passed to the compiler
(`workdir-$tag/Makefile`) and these changes will be picked up when running `make
simulation`.

### `make replay`

`svsim` by default emits an `execution-script.txt` when running a Simulation.
This script captures all commands sent to the simulation, which enables `svsim`
to add a second target to the `Makefile`: `make replay`. `make replay` rebuilds
the simulation, picking up any changes as mentioned above, and then replays the
commands that were sent by `Simulation.Controller` verbatim. This allows
replaying most tests without having to re-run the Scala code, potentially with
manual modifications to either the SystemVerilog or the test harness (though
tests which modify their behavior based on values read from the `Simulation` may
not be replayed faithfully).
