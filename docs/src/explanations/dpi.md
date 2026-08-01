---
layout: docs
title:  "SystemVerilog DPI ABI"
section: "chisel3"
---

# SystemVerilog DPI ABI

The `chisel3.util.circt.dpi` package provides intrinsic-based calls to
SystemVerilog Direct Programming Interface (DPI) functions. The `Raw*` call
objects create the `circt_dpi_call` intrinsic, which CIRCT lowers to an
`import "DPI-C"` declaration and a call to that function.

This page describes how Chisel types map to the generated SystemVerilog and
the C/C++ DPI ABI.

## Basic Example

```scala
import chisel3._
import chisel3.util.circt.dpi._

class DpiExample extends Module {
  val io = IO(new Bundle {
    val lhs = Input(UInt(32.W))
    val rhs = Input(UInt(32.W))
    val sum = Output(UInt(32.W))
  })

  val sum = RawUnclockedNonVoidFunctionCall(
    "add",
    UInt(32.W),
    Some(Seq("lhs", "rhs")),
    Some("result")
  )(true.B, io.lhs, io.rhs)

  io.sum := sum
}
```

The `UInt(32.W)` result is emitted as one SystemVerilog `output` argument:

```systemverilog
import "DPI-C" context function void add(
  input  int lhs,
  input  int rhs,
  output int result
);
```

The C/C++ function therefore returns `void` and writes the result through a
pointer:

```cpp
extern "C" void add(int lhs, int rhs, int *result) {
  *result = lhs + rhs;
}
```

`NonVoidFunctionCall` refers to the Chisel expression result. It does not mean
that the C function has a C return value.

## Scalar ABI

All integer values in DPI declarations use two-state SystemVerilog types.
The width determines the SystemVerilog type and how the value is passed to
C/C++:

| Chisel width | SystemVerilog type | C/C++ ABI |
| ---: | --- | --- |
| 1 | `bit` | passed by value as `svBit` |
| 8 | `byte` | passed by value |
| 16 | `shortint` | passed by value |
| 32 | `int` | passed by value |
| 64 | `longint` | passed by value |
| greater than 64 | `bit [W-1:0]` | passed by pointer using `svBitVecVal` |

The simulator-provided `svdpi.h` is authoritative for the DPI typedefs. Use
the corresponding C types from that header where available. In particular,
do not assume that a 64-bit `longint` corresponds to C `long` on every
platform; use the simulator's documented DPI prototype.

Widths must be known and must be 1, 8, 16, 32, 64, or greater than 64. Other
widths, such as 4 or 40, are rejected by the FIRRTL DPI intrinsic verifier.

Values wider than 64 bits are packed bit vectors passed by pointer. The pointer
refers to 32-bit `svBitVecVal` words as defined by `svdpi.h`; use the DPI
helpers from that header rather than treating the value as a native C integer.

The integer signedness of `UInt` and `SInt` does not select a different DPI
scalar type. Both are lowered to the corresponding two-state SystemVerilog
width. Apply any required signed or unsigned interpretation explicitly in
C/C++.

`Clock` and `enable` are intrinsic control operands. They do not appear in the
DPI declaration. Only the `data` operands become DPI input arguments.

## Arrays

A Chisel `Vec` is lowered to an unpacked open array at the DPI boundary.
For example, this raw intrinsic call uses 8-bit elements:

```scala
RawClockedVoidFunctionCall("consume_bytes", Some(Seq("values")))(
  clock,
  true.B,
  VecInit(Seq(1.U(8.W), 2.U(8.W)))
)
```

```systemverilog
import "DPI-C" context function void consume_bytes(
  input byte values[]
);
```

The C/C++ side receives an `svOpenArrayHandle`, not a raw `byte *`.
Use the standard open-array functions in `svdpi.h` to query and access it:

```cpp
#include "svdpi.h"

extern "C" void consume_bytes(const svOpenArrayHandle values) {
  const int size = svSize(values, 1);
  const auto *value = static_cast<const char *>(svGetArrElemPtr1(values, 0));
  // Use size and value here.
}
```

Use functions such as `svSize`, `svLow`, `svHigh`, and `svGetArrElemPtr1`.
Do not depend on the handle layout or element stride.

The element type follows the scalar rules. For example, `Vec(n, Bool())`
becomes `bit values[]`, `Vec(n, UInt(32.W))` becomes `int values[]`, and a
nested `Vec` produces nested unpacked array dimensions.

## Intrinsic Call Shapes

The intrinsic API has three call objects:

| Chisel call | DPI function shape |
| --- | --- |
| `RawClockedVoidFunctionCall` | `void f(input_args...)` |
| `RawClockedNonVoidFunctionCall` | `void f(input_args..., output_pointer)` |
| `RawUnclockedNonVoidFunctionCall` | `void f(input_args..., output_pointer)` |

The clocked and unclocked forms have the same ABI. They differ in scheduling:

- A clocked call is evaluated on the active edge of its supplied clock. Its
  result behaves as state and is retained when `enable` is false.
- An unclocked call is combinational. When `enable` is false, its result is
  undefined and should be treated as unknown.

`inputNames` supplies one SystemVerilog name for each input. `outputName` names
the one output argument of a non-void call. Neither changes the C/C++ ABI.

## Signature Rules

All intrinsic call sites using the same DPI function name must have matching
input and result types. CIRCT groups calls by function name and rejects
mismatched signatures during lowering.

The intrinsic requires passive operand and result types. Scalar integers and
`Vec` values have the ABI described above. Bundles lower to packed
SystemVerilog aggregates, but their direct C/C++ representation is not a
portable scalar ABI. Inspect the generated declaration and simulator DPI
prototype before using aggregate types.

For the underlying representation and lowering rules, see the
[FIRRTL DPI call intrinsic](https://circt.llvm.org/docs/Dialects/FIRRTL/#firrtlintdpicall-circtfirrtldpicallintrinsicop),
the [Simulation Dialect DPI](https://circt.llvm.org/docs/Dialects/SimDPI/),
and the [`LowerDPI` implementation](https://circt.llvm.org/doxygen/LowerDPI_8cpp_source.html).
