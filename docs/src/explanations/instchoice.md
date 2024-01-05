# Instance Choices

`Instance Choice`s are instances of modules whose targets are configurable post-elaboration.
They allow the target of an instance to be chosen from a pre-defined set after elaboration by
enabling an option through the ABI or through specialization in the compiler.

Instance choices rely on option groups to specify the available targets attached to each option:

```scala mdoc:silent
import chisel3.choice.{Case, Group}

object Platform extends Group {
  object FPGA extends Case
  object ASIC extends Case
}
```

The `Platform` option groups enumerates the list of platforms for which the design
can be specialised, such as `ASIC` or `FPGA`. Specialization is not mandatory: if an
option is left unspecified, a default variant is chosen.

The modules referenced by an instance choice must all specify the same IO interface by
deriving from `FixedIOBaseModule`. The `ModuleChoice` operator takes the default option
and a list of case-module mappings and returns a binding to the IO of the modules.

```scala mdoc:silent
import chisel3._
import chisel3.choice.ModuleChoice

class TargetIO extends Bundle {
  val in = Flipped(UInt(8.W))
  val out = UInt(8.W)
}

class FPGATarget extends FixedIOExtModule[TargetIO](new TargetIO)

class ASICTarget extends FixedIOExtModule[TargetIO](new TargetIO)

class VerifTarget extends FixedIORawModule[TargetIO](new TargetIO)

class SomeModule extends RawModule {
  val inst = ModuleChoice(new VerifTarget)(Seq(
    Platform.FPGA -> new FPGATarget,
    Platform.ASIC -> new ASICTarget
  ))
}
```
