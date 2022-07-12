---
layout: docs
title:  "Data Types"
section: "chisel3"
---

# Chisel Data Types

Chisel datatypes are used to specify the type of values held in state
elements or flowing on wires.  While hardware designs ultimately
operate on vectors of binary digits, other more abstract
representations for values allow clearer specifications and help the
tools generate more optimal circuits.  In Chisel, a raw collection of
bits is represented by the ```Bits``` type.  Signed and unsigned integers
are considered subsets of fixed-point numbers and are represented by
types ```SInt``` and ```UInt``` respectively. Signed fixed-point
numbers, including integers, are represented using two's-complement
format.  Boolean values are represented as type ```Bool```.  Note
that these types are distinct from Scala's builtin types such as
```Int``` or ```Boolean```.

> There is a new experimental type **Interval** which gives the developer more control of the type by allowing the definition of an IntervalRange.  See: [Interval Type](../appendix/experimental-features#interval-type)

Additionally, Chisel defines `Bundles` for making
collections of values with named fields (similar to ```structs``` in
other languages), and ```Vecs``` for indexable collections of
values.

Bundles and Vecs will be covered later.

Constant or literal values are expressed using Scala integers or
strings passed to constructors for the types:
```scala
1.U       // decimal 1-bit lit from Scala Int.
"ha".U    // hexadecimal 4-bit lit from string.
"o12".U   // octal 4-bit lit from string.
"b1010".U // binary 4-bit lit from string.

5.S    // signed decimal 4-bit lit from Scala Int.
-8.S   // negative decimal 4-bit lit from Scala Int.
5.U    // unsigned decimal 3-bit lit from Scala Int.

8.U(4.W) // 4-bit unsigned decimal, value 8.
-152.S(32.W) // 32-bit signed decimal, value -152.

true.B // Bool lits from Scala lits.
false.B
```
Underscores can be used as separators in long string literals to aid
readability, but are ignored when creating the value, e.g.:
```scala
"h_dead_beef".U   // 32-bit lit of type UInt
```

By default, the Chisel compiler will size each constant to the minimum
number of bits required to hold the constant, including a sign bit for
signed types. Bit widths can also be specified explicitly on
literals, as shown below. Note that (`.W` is used to cast a Scala Int
to a Chisel width)
```scala
"ha".asUInt(8.W)     // hexadecimal 8-bit lit of type UInt
"o12".asUInt(6.W)    // octal 6-bit lit of type UInt
"b1010".asUInt(12.W) // binary 12-bit lit of type UInt

5.asSInt(7.W) // signed decimal 7-bit lit of type SInt
5.asUInt(8.W) // unsigned decimal 8-bit lit of type UInt
```

For literals of type ```UInt```, the value is
zero-extended to the desired bit width.  For literals of type
```SInt```, the value is sign-extended to fill the desired bit width.
If the given bit width is too small to hold the argument value, then a
Chisel error is generated.

>We are working on a more concise literal syntax for Chisel using
symbolic prefix operators, but are stymied by the limitations of Scala
operator overloading and have not yet settled on a syntax that is
actually more readable than constructors taking strings.

>We have also considered allowing Scala literals to be automatically
converted to Chisel types, but this can cause type ambiguity and
requires an additional import.

>The SInt and UInt types will also later support an optional exponent
field to allow Chisel to automatically produce optimized fixed-point
arithmetic circuits.

## Casting

We can also cast types in Chisel:

```scala
val sint = 3.S(4.W)             // 4-bit SInt

val uint = sint.asUInt          // cast SInt to UInt
uint.asSInt                     // cast UInt to SInt
```

**NOTE**: `asUInt`/`asSInt` with an explicit width can **not** be used to cast (convert) between Chisel datatypes.
No width parameter is accepted, as Chisel will automatically pad or truncate as required when the objects are connected.

We can also perform casts on clocks, though you should be careful about this, since clocking (especially in ASIC) requires special attention:

```scala
val bool: Bool = false.B        // always-low wire
val clock = bool.asClock        // always-low clock

clock.asUInt                    // convert clock to UInt (width 1)
clock.asUInt.asBool             // convert clock to Bool (Chisel 3.2+)
clock.asUInt.toBool             // convert clock to Bool (Chisel 3.0 and 3.1 only)
```

## Analog/BlackBox type

(Experimental, Chisel 3.1+)

Chisel supports an `Analog` type (equivalent to Verilog `inout`) that can be used to support arbitrary nets in Chisel. This includes analog wires, tri-state/bi-directional wires, and power nets (with appropriate annotations).

`Analog` is an undirectioned type, and so it is possible to connect multiple `Analog` nets together using the `attach` operator. It is possible to connect an `Analog` **once** using `<>` but illegal to do it more than once.

```scala
val a = IO(Analog(1.W))
val b = IO(Analog(1.W))
val c = IO(Analog(1.W))

// Legal
attach(a, b)
attach(a, c)

// Legal
a <> b

// Illegal - connects 'a' multiple times
a <> b
a <> c
```

## Scala Type vs Chisel Type vs Hardware



The *Scala* type of the Data is recognized by the Scala compiler, such as `Decoupled[UInt]` or `MyBundle`, where 
```
MyBundle(w: Int) extends Bundle {val foo: UInt(w.W), val bar: UInt(w.W)}
```

The *Chisel* type of a Data is all the fields actually present, by names, and their types including widths. For example, `MyBundle(3)` creates a Chisel Type of `Record` with `foo : UInt(3.W),  bar: UInt(3.W))`.

A hardware is something that is "bound" to synthesizable hardware. For example `false.B` or `Reg(Bool())`.

A literal is a `Data` that is respresented as a literal value without being wrapped in Wire, Reg, or IO. 

The Scala compiler cannot distinguish between Chisel's representation of hardware `false.B`, `Reg(Bool())`
and pure Chisel types (e.g. `Bool()`). You can get runtime errors passing a Chisel type when hardware is expected, and vice versa.

```scala mdoc
import chisel3._
import chisel3.experimental.BundleLiterals._
import chisel3.stage.ChiselStage

class MyBundle(w: Int) extends Bundle {
    val foo = UInt(w.W)
    val bar = UInt(w.W) 
}

class MyModule(gen: () => MyBundle, demo: Int ) extends Module {
                                                     // Synthesizable   Literal
    val xType: MyBundle = new MyBundle(3)            //      -             -
    val x:     MyBundle = IO(Input(new MyBundle(3))) //      x             -       
    val xLit:  MyBundle = xType.Lit(                 //      -             - 
      _.foo -> 0.U(3.W), 
      _.bar -> 0.U(3.W)
    )
    //val y:   MyBundle = gen()                      //      ?             ?                      
 

    val foo: MyBundle = demo match { // foo is always synthesizable
      // These will work for either case:
      case 0 => 0.U.asTypeOf(gen())

      // If gen() is Synthesizable, these are allowed:
      case 1 => gen()
      case 2 => 0.U.asTypeOf(chiselTypeOf(gen()))    
      case 3 =>  Wire(chiselTypeOf(gen()))
      case 4 =>  WireInit(gen())

      // If unknown is a pure chisel type, these are allowed:
      case 5 => Wire(gen())
      case 6 => gen().Lit(_.foo -> 0.U, 
      _.bar -> 0.U )
      case 7 => {
        class Foo extends Bundle {
          val nested = gen()
        } 
      Wire(new Foo()).nested}

      // default
      case _ => Wire(new MyBundle(3))
    }

    if (!foo.isLit) {
      foo := DontCare
    }
}

class Wrapper(demo: Int, passSynthesizable: Boolean) extends Module {
  val gen = if (passSynthesizable) {
      () => { val gen = Wire(new MyBundle(3)) ; gen := DontCare; gen}
  } else (() => new MyBundle(3))
  val inst = Module(new MyModule(gen, demo))
  inst.x := DontCare
}
```

```scala mdoc:silent
// Work for both
ChiselStage.elaborate(new Wrapper(0, true))
ChiselStage.elaborate(new Wrapper(0, false))

// Only work if synthesizable
ChiselStage.elaborate(new Wrapper(1, true))
ChiselStage.elaborate(new Wrapper(2, true))
ChiselStage.elaborate(new Wrapper(3, true))
ChiselStage.elaborate(new Wrapper(4, true))

// Only work for Chisel Types
ChiselStage.elaborate(new Wrapper(5, false))
ChiselStage.elaborate(new Wrapper(6, false))
ChiselStage.elaborate(new Wrapper(7, false))
```

Can only `:=` to hardware:
```scala mdoc:crash
ChiselStage.elaborate(new Wrapper(1, false))
```


Have to pass hardware to chiselTypeOf:
```scala mdoc:crash
ChiselStage.elaborate(new Wrapper(2, false))
```
```scala mdoc:crash
ChiselStage.elaborate(new Wrapper(3, false))
```

Have to pass hardware to *Init:
```scala mdoc:crash
ChiselStage.elaborate(new Wrapper(4, false))
```


Can't pass hardware to a Wire, Reg, IO:
```scala mdoc:crash
ChiselStage.elaborate(new Wrapper(5, true))
```

.Lit can only be called on Chisel type:
```scala mdoc:crash
ChiselStage.elaborate(new Wrapper(6, true))
```

Can only use a Chisel type within a Bundle definition:
```scala mdoc:crash
ChiselStage.elaborate(new Wrapper(7, true))
```
