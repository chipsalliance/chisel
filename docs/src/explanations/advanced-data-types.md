---
layout: docs
title:  "Advanced Data Types"
section: "chisel3"
---

# Advanced Data Types

The Scala compiler cannot distinguish between Chisel's representation of hardware such as `false.B`, `Reg(Bool())`
and pure Chisel types (e.g. `Bool()`). You can get runtime errors passing a Chisel type when hardware is expected, and vice versa.

## Scala Type vs Chisel Type vs Hardware

The *Scala* type of the Data is recognized by the Scala compiler, such as `Decoupled[UInt]` or `MyBundle`, where 
```
MyBundle(w: Int) extends Bundle {val foo: UInt(w.W), val bar: UInt(w.W)}
```

The *Chisel* type of a Data is all the fields actually present, by names, and their types including widths and directions.
For example, `MyBundle(3)` creates a Chisel Type of `Record` with `foo : UInt(3.W),  bar: UInt(3.W))`.

Hardware is something that is "bound" to synthesizable hardware. For example `false.B` or `Reg(Bool())`.

A literal is a `Data` that is respresentable as a literal value without being wrapped in Wire, Reg, or IO. 

## Demo Code

The below code demonstrates how objects with the same Scala type can have different properties.

```scala mdoc
import chisel3._
import chisel3.experimental.BundleLiterals._
import chisel3.stage.ChiselStage

class MyBundle(w: Int) extends Bundle {
    val foo = UInt(w.W)
    val bar = UInt(w.W) 
}

class MyModule(gen: () => MyBundle, demo: Int) extends Module {
                                                            // Synthesizable   Literal
    val xType:    MyBundle     = new MyBundle(3)            //      -             -
    val dirXType: MyBundle     = Input(new MyBundle(3))     //      -             -
    val x:        MyBundle     = IO(Input(new MyBundle(3))) //      x             -       
    val xLit:     MyBundle     = xType.Lit(                 //      x             x 
      _.foo -> 0.U(3.W), 
      _.bar -> 0.U(3.W)
    )
    //val y:   MyBundle = gen()                             //      ?             ?                      
 

    // This code sets up a series of experiments which will either pass or fail
    // given whehter gen() is synthesizable or a plain Chisel type.
    val result: MyBundle = demo match { // result is always synthesizable
      // These will work for either case:
      case 0 => 0.U.asTypeOf(gen())

      // If gen() is Synthesizable, these are allowed:
      case 1 =>  gen()
      case 2 =>  0.U.asTypeOf(chiselTypeOf(gen()))    
      case 3 =>  Wire(chiselTypeOf(gen()))
      case 4 =>  WireInit(gen())

      // If gen is a pure chisel type, these are allowed:
      case 5 => Wire(gen())
      case 6 => gen().Lit(_.foo -> 0.U, 
      _.bar -> 0.U )
      case 7 => Wire(new Bundle {
        val nested = gen()
      }).nested

      // default
      case _ => Wire(new MyBundle(3))
    }

    if (!result.isLit) {
      result := DontCare
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

