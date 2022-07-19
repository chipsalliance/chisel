---
layout: docs
title:  "Chisel Type vs Scala Type"
section: "chisel3"
---

# Chisel Type vs Scala Type

The Scala compiler cannot distinguish between Chisel's representation of hardware such as `false.B`, `Reg(Bool())`
and pure Chisel types (e.g. `Bool()`). You can get runtime errors passing a Chisel type when hardware is expected, and vice versa.

## Scala Type vs Chisel Type vs Hardware

The *Scala* type of the Data is recognized by the Scala compiler, such as `Decoupled[UInt]` or `MyBundle` in 
```
MyBundle(w: Int) extends Bundle {val foo: UInt(w.W), val bar: UInt(w.W)}
```

The *Chisel* type of a `Data` is a Scala object. It captures all the fields actually present,
by names, and their types including widths.
For example, `MyBundle(3)` creates a Chisel Type of `Record` with `foo : UInt(3.W),  bar: UInt(3.W))`.

Hardware is something that is "bound" to synthesizable hardware. For example `false.B` or `Reg(Bool())`.
The binding is what determines the actual directionality of each field, it is not a property of the Chisel type.

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
}
```

## Chisel Type vs Hardware Type -- Specific Functions and Errors

`.asTypeOf` works for both hardware and Chisel type:

```scala mdoc:silent
ChiselStage.elaborate(new Module {
  val chiselType = new MyBundle(3)
  val hardware = Wire(new MyBundle(3))
  hardware := DontCare
  val a = 0.U.asTypeOf(chiselType)
  val b = 0.U.asTypeOf(hardware)
})
```

Can only `:=` to hardware:

```scala mdoc:silent
// Do this...
ChiselStage.elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  hardware := DontCare
})
```
```scala mdoc:crash
// Not this...
ChiselStage.elaborate(new Module {
  val chiselType = new MyBundle(3)
  chiselType := DontCare
})
```

Can only `:=` from hardware:
```scala mdoc:silent
// Do this...
ChiselStage.elaborate(new Module {
  val hardware = IO(new MyBundle(3))
  val moarHardware = Wire(new MyBundle(3))
  moarHardware := DontCare
  hardware <> moarHardware
})
```
```scala mdoc:crash
// Not this...
ChiselStage.elaborate(new Module {
  val hardware = IO(new MyBundle(3))
  val chiselType = new MyBundle(3)
  hardware <> chiselType
})
```

Have to pass hardware to `chiselTypeOf`:
```scala mdoc:silent
// Do this...
ChiselStage.elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  hardware := DontCare
  val chiselType = chiselTypeOf(hardware)
})
```
```scala mdoc:crash
// Not this...
ChiselStage.elaborate(new Module {
  val chiselType = new MyBundle(3)
  val crash = chiselTypeOf(chiselType)
})
```

Have to pass hardware to `*Init`:

```scala mdoc:silent
// Do this...
ChiselStage.elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  hardware := DontCare
  val moarHardware = WireInit(hardware)
})
```
```scala mdoc:crash
// Not this...
ChiselStage.elaborate(new Module {
  val crash = WireInit(new MyBundle(3))
})
```

Can't pass hardware to a `Wire`, `Reg`, `IO`:
```scala mdoc:silent
// Do this...
ChiselStage.elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  hardware := DontCare
})
```
```scala mdoc:crash
// Not this...
ChiselStage.elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  val crash = Wire(hardware)
})
```

`.Lit` can only be called on Chisel type:
```scala mdoc:silent
// Do this...
ChiselStage.elaborate(new Module {
  val hardwareLit = (new MyBundle(3)).Lit(
    _.foo -> 0.U, 
    _.bar -> 0.U
  )
})
```
```scala mdoc:crash
//Not this...
ChiselStage.elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  val crash = hardware.Lit(
    _.foo -> 0.U,
    _.bar -> 0.U
  )
})
```

Can only use a Chisel type within a `Bundle` definition:

```scala mdoc:silent
// Do this...
ChiselStage.elaborate(new Module {
  val hardware = Wire(new Bundle {
    val nested = new MyBundle(3)
  })
  hardware := DontCare
})
```
```scala mdoc:crash
// Not this...
ChiselStage.elaborate(new Module {
  val crash = Wire(new Bundle {
    val nested = Wire(new MyBundle(3))
  })
})
```

Can only call `directionOf` on Hardware:
```scala mdoc:silent
import chisel3.experimental.DataMirror

class Child extends Module{
  val hardware = IO(new MyBundle(3))
  hardware := DontCare
  val chiselType = new MyBundle(3)
}
```
```scala mdoc:silent
// Do this...
ChiselStage.elaborate(new Module {
  val child = Module(new Child())
  child.hardware := DontCare
  val direction = DataMirror.directionOf(child.hardware)
})
```
```scala mdoc:crash
// Not this...
ChiselStage.elaborate(new Module {
val child = Module(new Child())
  child.hardware := DontCare
  val direction = DataMirror.directionOf(child.chiselType)
})
```

Can call `specifiedDirectionOf` on hardware or Chisel type:

```scala mdoc:silent
ChiselStage.elaborate(new Module {
  val child = Module(new Child())
  child.hardware := DontCare
  val direction0 = DataMirror.specifiedDirectionOf(child.hardware)
  val direction1 = DataMirror.specifiedDirectionOf(child.chiselType)
})
```


```scala mdoc
//TODO:
// .asTypeOf vs .asInstanceOf (chisel type vs scala type)
// (new MyBundle(3)).asInstanceOf[Bundle]
// (gen(): Bundle).asInstanceOf[MyBundle]
// foo: Bundle = new MyBundle(3)
// someData.asTypeOf(someOtherData), chiselTypeOf(foo: Data), .asInstanceOf

// DataView stuff

// directionOf, specifiedDirectionOf

// type Erasure, isA, asInstanceOf

//.asInstanceOf[new MyBundle(3)]
//.asInstanceOf[MyBundle]
```