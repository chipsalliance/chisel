---
layout: docs
title:  "Calling Native Functions from Chisel (DPI)"
section: "chisel3"
---

# Calling Native Functions from Chisel (DPI)

## DPI Basics

Chisel's DPI API allows you to integrate native code into your Chisel hardware
designs.
This enables you to leverage existing libraries or implement functionality that
is difficult to express directly in Chisel.

The C/C++ implementation uses the `svdpi.h` header supplied by the simulator.
This header defines DPI types and helper functions such as `svBitVecVal` and
`svOpenArrayHandle`.
Chisel does not provide this header; the simulator's include path must be used
when compiling the native implementation.

Here's a simple example that demonstrates printing a message from a C++
function:

```c++
#include <iostream>

extern "C" void hello()
{
    std::cout << "hello from c++\n";
}
```

The `Raw*` API creates a `circt_dpi_call` intrinsic, which CIRCT lowers to an
`import "DPI-C"` declaration and a call to that function.

```scala mdoc:silent
import chisel3._
import chisel3.util.circt.dpi._

class HelloTest extends Module {
  RawClockedVoidFunctionCall("hello")(clock, true.B)
}
```

The call is placed inside a `Module` because it is a hardware intrinsic.
The generated SystemVerilog declaration is equivalent to:

```systemverilog
import "DPI-C" context function void hello();
```

## Type ABI

Unlike normal Chisel compilation flow, we use a specific ABI for types to
interact with DPI.

### Argument Types

* Operand and result types must be passive.
* A `Vec` is lowered to an *unpacked* *open* array type, e.g., `Vec(4, UInt(8.W))` to `byte values[]`.
* A `Bundle` is lowered to a packed struct.
* Integer values are lowered into two-state SystemVerilog types.

Here, passive means that the type contains no flipped or otherwise directioned
fields.
For example, a `Bundle` passed to DPI must not contain `Input` or `Output`
members.

The 8-, 16-, 32-, and 64-bit forms use SystemVerilog's `byte`, `shortint`,
`int`, and `longint` types and are passed by value using the corresponding
simulator DPI scalar types.
Users are required to use the specific integer widths shown in the table below.
Large integers are lowered to packed `bit` vectors and passed by pointer using
`svBitVecVal`.

| Width | SystemVerilog Type | Argument Passing Mode |
| ----- | ------------------ | --------------------- |
| 1     | `bit`              | value                 |
| 8     | `byte`             | value                 |
| 16    | `shortint`         | value                 |
| 32    | `int`              | value                 |
| 64    | `longint`          | value                 |
| > 64  | `bit [W-1:0]`      | pointer               |

Widths must be known and must be 1, 8, 16, 32, 64, or greater than 64.
Other widths, such as 4 or 40, are rejected by the FIRRTL DPI intrinsic
verifier.

The simulator-provided `svdpi.h` is authoritative for the DPI typedefs.
In particular, do not assume that a 64-bit `longint` corresponds to C `long`
on every platform.

The signedness of `UInt` and `SInt` does not select a different DPI scalar type.
Both are lowered to the corresponding two-state SystemVerilog width.
Apply any required signed or unsigned interpretation explicitly in C/C++.

### Function Types

There are several intrinsic call objects for DPI functions.
Which one you use depends on whether the call is void or produces a Chisel
result, and whether it is clocked or unclocked.

`RawClockedVoidFunctionCall` is used for a function with no Chisel result.
`RawClockedNonVoidFunctionCall` is used for a clocked function with one Chisel
result.
`RawUnclockedNonVoidFunctionCall` is used for an unclocked function with one
Chisel result.

A non-void Chisel result is emitted as the last SystemVerilog `output` argument.
It is not a C return value; the C function returns `void` and writes through a
pointer.

A clocked call is evaluated at the associated clock's positive edge.
Its result behaves as state and is retained when `enable` is false.
An unclocked call is combinational.
When `enable` is false, its result is undefined and should be treated as
unknown.

The clock and enable operands are intrinsic control operands.
They do not appear in the DPI declaration.
Only the `data` operands become DPI input arguments.

## Example: Adding Two Numbers

Here's an example of a DPI function that calculates the sum of two numbers.
The `result` pointer is required because the Chisel result is emitted as an
output argument rather than a C return value:

```c++
extern "C" void add(int lhs, int rhs, int* result)
{
    *result = lhs + rhs;
}
```

```scala mdoc:silent
class AddTest extends Module {
  val io = IO(new Bundle {
    val a = Input(UInt(32.W))
    val b = Input(UInt(32.W))
    val clocked = Output(UInt(32.W))
    val unclocked = Output(UInt(32.W))
    val enable = Input(Bool())
  })

  io.clocked := RawClockedNonVoidFunctionCall(
    "add",
    UInt(32.W),
    Some(Seq("lhs", "rhs")),
    Some("result")
  )(clock, io.enable, io.a, io.b)

  io.unclocked := RawUnclockedNonVoidFunctionCall(
    "add",
    UInt(32.W),
    Some(Seq("lhs", "rhs")),
    Some("result")
  )(io.enable, io.a, io.b)
}
```

```scala mdoc:verilog
chisel3.docs.emitSystemVerilog(new AddTest)
```

`inputNames` and `outputName` provide optional names for the SystemVerilog
arguments.
They do not change the C/C++ ABI.

## Example: Sum of an Array

Chisel vectors are converted into SystemVerilog open arrays when used with DPI.
Since memory layout can vary between simulators, use the standard open-array
functions from `svdpi.h` to access array elements.

```c++
#include "svdpi.h"

extern "C" void sum(const svOpenArrayHandle array, int* result) {
  const int size = svSize(array, 1);
  *result = 0;
  for (int i = 0; i < size; ++i) {
    svBitVecVal value;
    svGetBitArrElemVecVal(&value, array, i);
    *result += value;
  }
}
```

```scala mdoc:silent
class SumTest extends Module {
  val io = IO(new Bundle {
    val values = Input(Vec(3, UInt(32.W)))
    val result = Output(UInt(32.W))
  })

  io.result := RawUnclockedNonVoidFunctionCall(
    "sum",
    UInt(32.W),
    Some(Seq("array")),
    Some("result")
  )(true.B, io.values)
}
```

```scala mdoc:verilog
chisel3.docs.emitSystemVerilog(new SumTest)
```

The `Vec` input is declared as an unpacked open array, equivalent to:

```systemverilog
import "DPI-C" context function void sum(
  input int array[],
  output int result
);
```

The C/C++ side receives an `svOpenArrayHandle`, not a raw element pointer.
Use functions such as `svSize`, `svLow`, `svHigh`, and
`svGetBitArrElemVecVal` rather than depending on the handle layout or element
stride.

## FAQ

* Can Chisel export functions through this API? -- No, this API only creates calls to imported DPI functions.
  Consider using a black box for exported functionality.
* Can we call a DPI function in an initial block? -- No, not currently.
  Consider using a black box for initialization.
* Can we call two clocked DPI calls and pass the result of one to the other within the same clock? -- No.
  Do not rely on ordering between clocked DPI calls on the same clock.
  Combine dependent operations into one DPI function when ordering matters.

For the underlying representation and lowering rules, see the
[FIRRTL DPI call intrinsic](https://circt.llvm.org/docs/Dialects/FIRRTL/#firrtlintdpicall-circtfirrtldpicallintrinsicop),
the [Simulation Dialect DPI](https://circt.llvm.org/docs/Dialects/SimDPI/),
and the [`LowerDPI` implementation](https://circt.llvm.org/doxygen/LowerDPI_8cpp_source.html).
