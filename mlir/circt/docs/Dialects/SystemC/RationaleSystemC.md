# SystemC Dialect Rationale

This document describes various design points of the SystemC dialect, why they
are the way they are, and current status. This follows in the spirit of other
[MLIR Rationale docs](https://mlir.llvm.org/docs/Rationale/).

- [SystemC Dialect Rationale](#systemc-dialect-rationale)
  - [Introduction](#introduction)
  - [Lowering](#lowering)
  - [Q&A](#qa)


## Introduction

[SystemC](https://en.wikipedia.org/wiki/SystemC) is a library written in C++
to allow functional modeling of systems. The included event-driven simulation
kernel can then be used to simulate a system modeled entirely in SystemC.
Additionally, SystemC is a standard (IEEE Std 1666-2011) supported by several
tools (e.g., Verilator) and can thus be used as an interface to such tools as
well as between multiple systems that are internally using custom
implementations.

Enabling CIRCT to emit SystemC code provides another way (next to Verilog
emission) to interface with the outside-world and at the same time
provides another way to simulate systems compiled with CIRCT.

## Lowering

In a first step, lowering from [HW](https://circt.llvm.org/docs/Dialects/HW/)
to the SystemC dialect will be implemented. A tool called ExportSystemC,
which is analogous to ExportVerilog, will then take these SystemC and
[Comb](https://circt.llvm.org/docs/Dialects/Comb/) operations to emit proper
SystemC-C++ code that can be compiled using clang, GCC, or any other
C++-compiler to produce the simulator binary. In the long run support for more
dialects can be added, such as LLHD and SV.

As a simple example we take a look at the following HW module which just adds
two numbers together:

```mlir
hw.module @adder (%a: i32, %b: i32) -> (c: i32) {
    %sum = comb.add %a, %b : i32
    hw.output %sum : i32
}
```

It will then be lowered to the following SystemC IR to make code emission
easier for ExportSystemC:

```mlir
systemc.module @adder(%a: i32, %b: i32) -> (%c: i32) {
    systemc.ctor {
        systemc.method @add
    }
    systemc.func @add() {
        // c = a + b
        %res = comb.add %a, %b : i32
        systemc.con %c, %res : i32
    }
}
```

ExportSystemC will then emit the following C++ code to be compiled by clang or
another C++-compiler:

```cpp
#ifndef ADDER_H
#define ADDER_H

#include <systemc.h>

SC_MODULE(adder) {
    sc_in<sc_uint<32>> a;
    sc_in<sc_uint<32>> b;
    sc_out<sc_uint<32>> c;

    SC_CTOR(adder) {
        SC_METHOD(add);
    }

    void add() {
        c = a + b;
    }
};

#endif // ADDER_H
```


## Q&A

**Q: Why implement a custom module operation rather than using `hw.module`?**

In SystemC we want to model module outputs as arguments such that the SSA value
is already defined from the beginning which we can then assign to and reference.

**Q: Why implement a custom func operation rather than using `func.func`?**

An important difference compared to the `func.func` operation is that it
represents a member function (method) of a SystemC module, i.e., a C++ struct.
This leads to some implementation differences:
* Not isolated from above: we need to be able to access module fields such as
  the modules inputs, outputs, and signals
* Verified to have no arguments and void return type: this is a restriction
  from SystemC for the function to be passed to SC_METHOD, etc. This could 
  also be achieved with `func.func`, but would require us to write the verifier
  in `systemc.module` instead.
* Region with only a single basic block (structured control flow) and no
  terminator: using structured control-flow leads to easier code emission

**Q: How much of C++ does the SystemC dialect aim to model?**

As much as necessary, as little as possible. Completely capturing C++ in a
dialect would be a huge undertaking and way too much to 'just' achieve SystemC
emission. At the same time, it is not possible to not model any C++ at all,
because when only modeling SystemC specific constructs, the gap for
ExportSystemC to bridge would be too big (we want the printer to be as simple
as possible).

**Q: Why does `systemc.module` have a graph region rather than a SSACFG region?**

It contains a single graph region to allow flexible positioning of the fields,
constructor and methods to support different ordering styles (fields at top
or bottom, methods to be registered with SC_METHOD positioned after the
constructor, etc.) without requiring any logic in ExportSystemC. Program code
to change the emission style can thus be written as part of the lowering from
HW, as a pre-emission transformation, or anywhere else.
