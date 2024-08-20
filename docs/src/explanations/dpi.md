---
layout: docs
title:  "DPI"
section: "chisel3"
---

# DPI

# Hello world

Let's start with a DPI function that just prints a message in DPI library:
```c++
extern "C" void hello()
{
    std::cout << "hello from c++\\n";
}
```

We must select appropriate abstract DPI class based on (1) whether the function returns a value and (2) when to call the function:

```scala
object Hello extends DPIVoidFunctionImport {
  override val functionName = "hello"
  final def apply() = super.call()
}


Hello()
```

`hello` is a nullary void function so we must choose `DPIVoidFunctionImport` as a base class of function object. `DPIVoidFunctionImport` has abstract member `functionName` which is a c-linkage name users must provide. `DPIVoidFunctionImport` has a `call` method which internally generates
raw DPI intrinsic but it's recommended to define `apply` function so the object looks exactly 
scala functions. 

## Types
System Verilog defines c-compatible types

## Non-void function
If the function returns a value we must use `DPINonVoidFunctionImport[T]` where `T` is a return type. For example let's consider `Add` function that calculates sum 

```scala
object Add extends DPINonVoidFunctionImport[Unit] {
  override val functionName = "add"
  override val ret = UInt(32.W)
  override val clocked = true
  override val inputNames = Some(Seq("lhs", "rhs"))
  override val outputName = Some("result")
  final def apply(lhs: UInt, rhs: UInt): UInt = super.call(lhs, rhs)
}
```

## Clocked vs unclocked function call


## Open Array 
Chisel Vector is always lowered into an open array. 

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

val sum_a = Sum(io.a)
val sum_b = Sum(io.a)
``

# FAQ

## Can we export functions?
Not yet. Please use a blackbox at this point.

## Can we call DPI function in initial block?
Not yet. Please use a blackbox at this point.

## Can we represent dependency between two DPI calls?
