---
layout: docs
title:  "Calling Native Functions from Chisel (DPI)"
section: "chisel3"
---

# Calling Native Functions from Chisel (DPI)

## DPI Basics

DPI API allows you to integrate native code into your Chisel hardware designs. This enables you to leverage existing libraries or implement functionality that is difficult to express directly in Chisel.

Here's a simple example that demonstrates printing a message from a C++ function:
```c++
extern "C" void hello()
{
    std::cout << "hello from c++\\n";
}
```

To call this function from Chisel, we need to define a corresponding DPI object.

```scala
object Hello extends DPIVoidFunctionImport {
  override val functionName = "hello"
  override val clocked = true
  final def apply() = super.call()
}

Hello() // Print
```

Explanation:

* `Hello` inherits from `DPIVoidFunctionImport` because the C++ function doesn't return a value (void).
* `functionName` specifies the C-linkage name of the C++ function.
* `clocked = true` indicates that its function call is invoked at clock's posedge.
* We recommend defining the apply method for a more Scala-like syntax.

## Type ABI

Unlike normal Chisel compilation flow, we use a specific ABI for types to interact with DPI.

### Argument Types

* Operand and result types must be passive.
* A vector is lowered to an *unpacked* *open* array type, e.g., `a: Vec<4, UInt>` to `byte a []`.
* A bundle is lowered to a packed struct.
* Integer types are lowered into 2-state types.
Small integer types (< 64 bit) must be compatible with C-types and arguments are passed by value. Users are required to use specific integer types for small integers shown in the table below. Large integers are lowered to bit and passed by reference.


| Width | Verilog Type | Argument Passing Modes |
| ----- | ------------ | ---------------------- |
| 1     | bit          | value                  |
| 8     | byte         | value                  |
| 16    | shortint     | value                  |
| 32    | int          | value                  |
| 64    | longint      | value                  |
| > 64  | bit [w-1:0]  | reference              |

### Function Types
The type of DPI object you need depends on the characteristics of your C++ function:

* Return Type
  * For functions that don't return a value (like hello), use `DPIVoidFunctionImport`.
  * For functions with a return type (e.g., integer addition), use `DPINonVoidFunctionImport[T]`, where `T` is the return type.
  * The output argument must be the last argument in DPI fuction.
* Clocked vs. Unclocked:
  * Clocked: The function call is evaluated at the associated clock's positive edge. By default, Chisel uses the current module's clock. For custom clocks, use `withClocked(clock) {..}`. If the function has a return value, it will be available in the next clock cycle.
  * Unclocked: The function call is evaluated immediately.

## Example: Adding Two Numbers
Here's an example of a DPI function that calculates the sum of two numbers:

```c++
extern "C" void add(int lhs, int rhs, int* result)
{
    *result = lhs + rhs;
}
```

```scala
object Add extends DPINonVoidFunctionImport[UInt] {
  override val functionName = "add"
  override val ret = UInt(32.W)
  override val clocked = false
  override val inputNames = Some(Seq("lhs", "rhs"))
  override val outputName = Some("result")
  final def apply(lhs: UInt, rhs: UInt): UInt = super.call(lhs, rhs)
}

val result = Add(2.U(32.W), 3.U(32.W))
```

Explanation:

* `Add` inherits from `DPINonVoidFunctionImport[UInt]` because it returns a 32-bit unsigned integer.
* `ret` specifies the return type.
* `clocked` indicates that this is a clocked function call.
* `inputNames` and `outputName` provide optional names for the function's arguments and return value (these are just for Verilog readability).

## Example: Sum of an array
Chisel vectors are converted into SystemVerilog open arrays when used with DPI. Since memory layout can vary between simulators, it's recommended to use `svSize` and `svGetBitArrElemVecVal` to access array elements.

```c++
extern "C" void sum(const svOpenArrayHandle array, int* result) {
  // Get a length of the open array.
  int size = svSize(array, 1);
  // Initialize the result value.
  *result = 0;
  for(size_t i = 0; i < size; ++i) {
    svBitVecVal vec;
    svGetBitArrElemVecVal(&vec, array, i);
    *result += vec;
  }
}
```

```scala
object Sum extends DPINonVoidFunctionImport[UInt] {
  override val functionName = "sum"
  override val ret = UInt(32.W)
  override val clocked = false
  override val inputNames = Some(Seq("array"))
  override val outputName = Some("result")
  final def apply(array: Vec[UInt]): UInt = super.call(array)
}

val io = IO(new Bundle {
  val a = Input(Vec(3, UInt(32.W)))
  val b = Input(Vec(6, UInt(32.W)))
})

val sum_a = Sum(io.a) // compute a[0] + a[1] + a[2]
val sum_b = Sum(io.b) // compute b[0] + ... + b[5]
```

# FAQ

* Can we export functions? -- No, not currently. Consider using a black box for such functionality.
* Can we call a DPI function in initial block? -- No, not currently. Consider using a black box for initialization.
* Can we call two clocked DPI calls and pass the result to another within the same clock? -- No, not currently. Please merge the DPI functions into a single function.
