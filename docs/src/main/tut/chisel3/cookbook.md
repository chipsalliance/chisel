---
layout: docs
title:  "Cookbook"
section: "chisel3"
---

Welcome to the Chisel cookbook. This cookbook is still in early stages. If you have any requests or examples to share, please [file an issue](https://github.com/ucb-bar/chisel3/issues/new) and let us know!

Please note that these examples make use of [Chisel's scala-style printing](Printing-in-Chisel#scala-style).

* Converting Chisel Types to/from UInt
  * [How do I create a UInt from an instance of a Bundle?](#how-do-i-create-a-uint-from-an-instance-of-a-bundle)
  * [How do I create a Bundle from a UInt?](#how-do-i-create-a-bundle-from-a-uint)
  * [How do I create a Vec of Bools from a UInt?](#how-do-i-create-a-vec-of-bools-from-a-uint)
  * [How do I create a UInt from a Vec of Bool?](#how-do-i-create-a-uint-from-a-vec-of-bool)
* Vectors and Registers
  * [How do I create a Vector of Registers?](#how-do-i-create-a-vector-of-registers)
  * [How do I create a Reg of type Vec?](#how-do-i-create-a-reg-of-type-vec)
* [How do I create a finite state machine?](#how-do-i-create-a-finite-state-machine)
* [How do I unpack a value ("reverse concatenation") like in Verilog?](#how-do-i-unpack-a-value-reverse-concatenation-like-in-verilog)
* [How do I do subword assignment (assign to some bits in a UInt)?](#how-do-i-do-subword-assignment-assign-to-some-bits-in-a-uint)
* [How can I dynamically set/parametrize the name of a module?](#how-can-i-dynamically-setparametrize-the-name-of-a-module)
* [How do I create an optional I/O?](#how-do-i-create-an-optional-io)
* [How do I get Chisel to name signals properly in blocks like when/withClockAndReset?](#how-do-i-get-chisel-to-name-signals-properly-in-blocks-like-whenwithclockandreset)

## Converting Chisel Types to/from UInt

### How do I create a UInt from an instance of a Bundle?

Call asUInt on the Bundle instance.

```scala
  // Example
  class MyBundle extends Bundle {
    val foo = UInt(4.W)
    val bar = UInt(4.W)
  }
  val bundle = Wire(new MyBundle)
  bundle.foo := 0xc.U
  bundle.bar := 0x3.U
  val uint = bundle.asUInt
  printf(p"$uint") // 195

  // Test
  assert(uint === 0xc3.U)
```

### How do I create a Bundle from a UInt?

On an instance of the Bundle, call the method fromBits with the UInt as the argument

```scala
  // Example
  class MyBundle extends Bundle {
    val foo = UInt(4.W)
    val bar = UInt(4.W)
  }
  val uint = 0xb4.U
  val bundle = (new MyBundle).fromBits(uint)
  printf(p"$bundle") // Bundle(foo -> 11, bar -> 4)

  // Test
  assert(bundle.foo === 0xb.U)
  assert(bundle.bar === 0x4.U)
```

### How do I create a Vec of Bools from a UInt?

Use the builtin function chisel3.core.Bits.toBools to create a Scala Seq of Bool,
then wrap the resulting Seq in Vec(...)

```scala
  // Example
  val uint = 0xc.U
  val vec = Vec(uint.toBools)
  printf(p"$vec") // Vec(0, 0, 1, 1)

  // Test
  assert(vec(0) === false.B)
  assert(vec(1) === false.B)
  assert(vec(2) === true.B)
  assert(vec(3) === true.B)
```

### How do I create a UInt from a Vec of Bool?

Use the builtin function asUInt

```scala
  // Example
  val vec = Vec(true.B, false.B, true.B, true.B)
  val uint = vec.asUInt
  printf(p"$uint") // 13

  /* Test
   *
   * (remember leftmost Bool in Vec is low order bit)
   */
  assert(0xd.U === uint)
```

## Vectors and Registers

### How do I create a Vector of Registers?

**Rule!  Use Reg of Vec not Vec of Reg!**

You create a [Reg of type Vec](#how-do-i-create-a-reg-of-type-vec). Because Vecs are a *type* (like `UInt`, `Bool`) rather than a *value*, we must bind the Vec to some concrete *value*.

### How do I create a Reg of type Vec?

For information, please see the API documentation
(https://chisel.eecs.berkeley.edu/api/index.html#chisel3.core.Vec)

```scala
  // Reg of Vec of 32-bit UInts without initialization
  val regOfVec = Reg(Vec(4, UInt(32.W)))
  regOfVec(0) := 123.U // a couple of assignments
  regOfVec(2) := regOfVec(0)

  // Reg of Vec of 32-bit UInts initialized to zero
  //   Note that Seq.fill constructs 4 32-bit UInt literals with the value 0
  //   VecInit(...) then constructs a Wire of these literals
  //   The Reg is then initialized to the value of the Wire (which gives it the same type)
  val initRegOfVec = RegInit(VecInit(Seq.fill(4)(0.U(32.W))))

  // Simple test (cycle comes from superclass)
  when (cycle === 2.U) { assert(regOfVec(2) === 123.U) }
  for (elt <- initRegOfVec) { assert(elt === 0.U) }
```

### How do I create a finite state machine?

Use Chisel Enum to construct the states and switch & is to construct the FSM
control logic.

```scala
import chisel3._
import chisel3.util._

class DetectTwoOnes extends Module {
  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
  })

  val sNone :: sOne1 :: sTwo1s :: Nil = Enum(3)
  val state = RegInit(sNone)

  io.out := (state === sTwo1s)

  switch (state) {
    is (sNone) {
      when (io.in) {
        state := sOne1
      }
    }
    is (sOne1) {
      when (io.in) {
        state := sTwo1s
      } .otherwise {
        state := sNone
      }
    }
    is (sTwo1s) {
      when (!io.in) {
        state := sNone
      }
    }
  }
}
```

The `is` statement can take multiple conditions e.g. `is (sTwo1s, sOne1) { ... }`.

### How do I unpack a value ("reverse concatenation") like in Verilog?

```verilog
wire [1:0] a;
wire [3:0] b;
wire [2:0] c;
wire [8:0] z = [...];
assign {a,b,c} = z;
```

Unpacking often corresponds to reinterpreting an unstructured data type as a structured data type. Frequently, this structured type is used prolifically in the design, and has been declared as in the following example:

```scala
class MyBundle extends Bundle {
  val a = UInt(2.W)
  val b = UInt(4.W)
  val c = UInt(3.W)
}
```

The easiest way to accomplish this in Chisel would be:

```scala
val z = Wire(UInt(9.W))
// z := ...
val unpacked = z.asTypeOf(new MyBundle)
unpacked.a
unpacked.b
unpacked.c
```

If you **really** need to do this for a one-off case (Think thrice! It is likely you can better structure the code using bundles), then rocket-chip has a [Split utility](https://github.com/freechipsproject/rocket-chip/blob/723af5e6b69e07b5f94c46269a208a8d65e9d73b/src/main/scala/util/Misc.scala#L140) which can accomplish this.

### How do I do subword assignment (assign to some bits in a UInt)?

Example:
```scala
class TestModule extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(10.W))
    val bit = Input(Bool())
    val out = Output(UInt(10.W))
  })
  io.out(0) := io.bit
}
```

Chisel3 does not support subword assignment. We find that this type of thing can usually be better expressed with aggregate/structured types: Bundles and Vecs.

If you must express it this way, you can blast your UInt to a Vec of Bools and back:

```scala
class TestModule extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(10.W))
    val bit = Input(Bool())
    val out = Output(UInt(10.W))
  })
  val bools = VecInit(io.in.toBools)
  bools(0) := io.bit
  io.out := bools.asUInt
}
```

### How can I dynamically set/parametrize the name of a module?

You can override the `desiredName` function. This works with normal Chisel modules and `BlackBox`es. Example:
```scala
class Coffee extends BlackBox {
    val io = IO(new Bundle {
        val I = Input(UInt(32.W))
        val O = Output(UInt(32.W))
    })
    override def desiredName = "Tea"
}
class Salt extends Module {
    val io = IO(new Bundle {})
    val drink = Module(new Coffee)
    override def desiredName = "SodiumMonochloride"
}
```

Elaborating the Chisel module `Salt` yields:
```verilog
module SodiumMonochloride(
  input   clock,
  input   reset
);
  wire [31:0] drink_O;
  wire [31:0] drink_I;
  Tea drink (
    .O(drink_O),
    .I(drink_I)
  );
  assign drink_I = 32'h0;
endmodule
```

### How do I create an optional I/O?

The following example is a module which includes the optional port `out2` only if the given parameter is `true`.

```scala
class ModuleWithOptionalIOs(flag: Boolean) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(12.W))
    val out = Output(UInt(12.W))
    val out2 = if (flag) Some(Output(UInt(12.W))) else None
  })

  io.out := io.in
  if (flag) {
    io.out2.get := io.in
  }
}
```

### How do I get Chisel to name signals properly in blocks like when/withClockAndReset?

To get Chisel to name signals (wires and registers) declared inside of blocks like `when`, `withClockAndReset`, etc, use the `chiselName` annotation as shown below:

```scala
import chisel3._
import chisel3.experimental.chiselName

@chiselName
class TestMod extends Module {
  val io = IO(new Bundle {
    val a = Input(Bool())
    val b = Output(UInt(4.W))
  })
  when (io.a) {
    val innerReg = RegInit(5.U(4.W))
    innerReg := innerReg + 1.U
    io.b := innerReg
  } .otherwise {
    io.b := 10.U
  }
}
```

Note that you will need to add the following line to your project's `build.sbt` file.

```
addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)
```

Without `chiselName`, Chisel is not able to name `innerReg` correctly (notice the `_T`):

```verilog
module TestMod(
  input        clock,
  input        reset,
  input        io_a,
  output [3:0] io_b
);
  reg [3:0] _T;
  wire [3:0] _T_2;
  assign _T_2 = _T + 4'h1;
  assign io_b = io_a ? _T : 4'ha;
  always @(posedge clock) begin
    if (reset) begin
      _T <= 4'h5;
    end else begin
      _T <= _T_2;
    end
  end
endmodule
```

In contrast, Chisel is able to name `innerReg` correctly with `chiselName`:

```verilog
module TestMod(
  input        clock,
  input        reset,
  input        io_a,
  output [3:0] io_b
);
  reg [3:0] innerReg;
  wire [3:0] _T_1;
  assign _T_1 = innerReg + 4'h1;
  assign io_b = io_a ? innerReg : 4'ha;
  always @(posedge clock) begin
    if (reset) begin
      innerReg <= 4'h5;
    end else begin
      innerReg <= _T_1;
    end
  end
endmodule
```
