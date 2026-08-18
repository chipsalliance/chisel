# Chisel DPI

```scala mdoc:invisible
import chisel3._
import chisel3.util.circt.dpi._
```

Chisel provides intrinsics that call C functions during simulation through SystemVerilog DPI-C.
You can implement logic in C/C++ and call it from a Chisel design.

The [API documentation](https://www.chisel-lang.org/api/latest/chisel3/util/circt/dpi/) gives more details.

## Overview

A DPI function call from Chisel has two parts:

1. **The Chisel call site** declares the function name, argument types, return type, and clocking behavior.
2. **The C implementation** is an `extern "C"` function that the simulator compiles and links.

## Clocked vs. Unclocked

Each DPI call is clocked or unclocked.
The call type determines when the function runs and how its result behaves:

- **Clocked**: The function runs on the positive edge of `clock` when `enable` is high.
  The return value keeps its previous value when `enable` is low.
- **Unclocked**: The function evaluates combinationally when an input changes.
- **Void**: Void calls are always clocked.
  The function runs for its side effects on the positive edge when `enable` is high.

## Ad-hoc calling style

The ad-hoc intrinsics use the surrounding module's `clock`, `enable`, and hardware nodes.
The following code is in a host module that provides them:

```scala mdoc:silent
class DpiHost extends Module {
  val enable = IO(Input(Bool()))
  val a = IO(Input(UInt(32.W)))
  val b = IO(Input(UInt(32.W)))

  // Call "hello" on every posedge of clock where enable is high
  RawClockedVoidFunctionCall("hello")(clock, enable)

  // `RawClockedNonVoidFunctionCall`: clocked, registers its result
  val result = RawClockedNonVoidFunctionCall(
    "add",                        // Name of the C function to call
    UInt(32.W),                   // Chisel type of the return value
    Some(Seq("lhs", "rhs")),      // Names for the SV input parameters
    Some("result")                // Names for the SV output parameter
  )(clock, enable, a, b)          // clock, enable, then the data arguments
  // result updates on posedge clock; holds its value when enable is low

  // `RawUnclockedNonVoidFunctionCall`: no clock — result updates combinationally
  val unclockedResult = RawUnclockedNonVoidFunctionCall(
    "add",
    UInt(32.W),
    Some(Seq("lhs", "rhs")),
    Some("result")
  )(enable, a, b)  // no clock argument, just enable and data inputs
}
```

The three ad-hoc intrinsics are `RawClockedVoidFunctionCall`, `RawClockedNonVoidFunctionCall`, and `RawUnclockedNonVoidFunctionCall`.

## Object-oriented calling style

If you call a function from multiple locations, wrap it in a Scala object that extends a DPI trait.

The traits use `Module.clock`.
They do not need an explicit clock argument.

### `DPIClockedVoidFunctionImport`

```scala mdoc
// Wrap a void DPI function as a Scala object with an apply method
object Hello extends DPIClockedVoidFunctionImport {
  override val functionName = "hello"  // maps to extern "C" void hello()

  // call() invokes the function
  final def apply() = call()
}
```

### `DPINonVoidFunctionImport[T]`

```scala mdoc
// Wrap a non-void DPI function; type parameter is the return Chisel type
object Add extends DPINonVoidFunctionImport[UInt] {
  override val functionName = "add"
  override val ret = UInt(32.W)                      // determines the SV output type
  override val clocked = true                        // result latches on clock edge
  override val inputNames = Some(Seq("lhs", "rhs"))  // names in generated import declaration
  override val outputName = Some("result")

  // call() invokes the function and returns the hardware value
  final def apply(lhs: UInt, rhs: UInt): UInt =
    call(lhs, rhs)
}
```

## C Implementation

Declare DPI functions as `extern "C"`.
Inputs pass by value.
For non-void functions, add a pointer argument for the return value after the inputs.

```c
#include <stdint.h>

// Void function
extern "C" void hello() {
    printf("hello from c++\n");
}

// Non-void function with inputs and outputs
extern "C" void add(int lhs, int rhs, int* result) {
    *result = lhs + rhs;
}
```
