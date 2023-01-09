---
layout: docs
title:  "Chisel Type vs Scala Type"
section: "chisel3"
---

# Chisel Type vs Scala Type

The Scala compiler cannot distinguish between Chisel's representation of hardware such as `false.B`, `Reg(Bool())`
and pure Chisel types (e.g. `Bool()`). You can get runtime errors passing a Chisel type when hardware is expected, and vice versa.

## Scala Type vs Chisel Type vs Hardware

```scala mdoc:invisible
import chisel3._
import circt.stage.ChiselStage
```

The *Scala* type of the Data is recognized by the Scala compiler, such as `Bool`, `Decoupled[UInt]` or `MyBundle` in
```scala mdoc:silent
class MyBundle(w: Int) extends Bundle {
  val foo = UInt(w.W)
  val bar = UInt(w.W)
}
```

The *Chisel* type of a `Data` is a Scala object. It captures all the fields actually present,
by names, and their types including widths.
For example, `MyBundle(3)` creates a Chisel Type with fields `foo: UInt(3.W),  bar: UInt(3.W))`.

Hardware is `Data` that is "bound" to synthesizable hardware. For example `false.B` or `Reg(Bool())`.
The binding is what determines the actual directionality of each field, it is not a property of the Chisel type.

A literal is a `Data` that is respresentable as a literal value without being wrapped in Wire, Reg, or IO.

## Chisel Type vs Hardware vs Literals

The below code demonstrates how objects with the same Scala type (`MyBundle`) can have different properties.

```scala mdoc:silent
import chisel3.experimental.BundleLiterals._

class MyModule(gen: () => MyBundle) extends Module {
                                                            //   Hardware   Literal
    val xType:    MyBundle     = new MyBundle(3)            //      -          -
    val dirXType: MyBundle     = Input(new MyBundle(3))     //      -          -
    val xReg:     MyBundle     = Reg(new MyBundle(3))       //      x          -
    val xIO:      MyBundle     = IO(Input(new MyBundle(3))) //      x          -
    val xRegInit: MyBundle     = RegInit(xIO)               //      x          -
    val xLit:     MyBundle     = xType.Lit(                 //      x          x
      _.foo -> 0.U(3.W),
      _.bar -> 0.U(3.W)
    )
    val y:        MyBundle = gen()                          //      ?          ?

    // Need to initialize all hardware values
    xReg := DontCare
}
```

```scala mdoc:invisible
// Just here to compile check the above
def elaborate(module: => chisel3.RawModule) = {
  (new chisel3.stage.phases.Elaborate)
    .transform(Seq(chisel3.stage.ChiselGeneratorAnnotation(() => module)))
    .collectFirst { case chisel3.stage.ChiselCircuitAnnotation(circuit) =>
      circuit
    }
    .get
}
elaborate(new MyModule(() => new MyBundle(3)))
```

## Chisel Type vs Hardware -- Specific Functions and Errors

`.asTypeOf` works for both hardware and Chisel type:

```scala mdoc:silent
elaborate(new Module {
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
elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  hardware := DontCare
})
```
```scala mdoc:crash
// Not this...
elaborate(new Module {
  val chiselType = new MyBundle(3)
  chiselType := DontCare
})
```

Can only `:=` from hardware:
```scala mdoc:silent
// Do this...
elaborate(new Module {
  val hardware = IO(new MyBundle(3))
  val moarHardware = Wire(new MyBundle(3))
  moarHardware := DontCare
  hardware := moarHardware
})
```
```scala mdoc:crash
// Not this...
elaborate(new Module {
  val hardware = IO(new MyBundle(3))
  val chiselType = new MyBundle(3)
  hardware := chiselType
})
```

Have to pass hardware to `chiselTypeOf`:
```scala mdoc:silent
// Do this...
elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  hardware := DontCare
  val chiselType = chiselTypeOf(hardware)
})
```
```scala mdoc:crash
// Not this...
elaborate(new Module {
  val chiselType = new MyBundle(3)
  val crash = chiselTypeOf(chiselType)
})
```

Have to pass hardware to `*Init`:
```scala mdoc:silent
// Do this...
elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  hardware := DontCare
  val moarHardware = WireInit(hardware)
})
```
```scala mdoc:crash
// Not this...
elaborate(new Module {
  val crash = WireInit(new MyBundle(3))
})
```

Can't pass hardware to a `Wire`, `Reg`, `IO`:
```scala mdoc:silent
// Do this...
elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  hardware := DontCare
})
```
```scala mdoc:crash
// Not this...
elaborate(new Module {
  val hardware = Wire(new MyBundle(3))
  val crash = Wire(hardware)
})
```

`.Lit` can only be called on Chisel type:
```scala mdoc:silent
// Do this...
elaborate(new Module {
  val hardwareLit = (new MyBundle(3)).Lit(
    _.foo -> 0.U,
    _.bar -> 0.U
  )
})
```
```scala mdoc:crash
//Not this...
elaborate(new Module {
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
elaborate(new Module {
  val hardware = Wire(new Bundle {
    val nested = new MyBundle(3)
  })
  hardware := DontCare
})
```
```scala mdoc:crash
// Not this...
elaborate(new Module {
  val crash = Wire(new Bundle {
    val nested = Wire(new MyBundle(3))
  })
})
```

Can only call `directionOf` on Hardware:
```scala mdoc:silent
import chisel3.reflect.DataMirror

class Child extends Module{
  val hardware = IO(new MyBundle(3))
  hardware := DontCare
  val chiselType = new MyBundle(3)
}
```
```scala mdoc:silent
// Do this...
elaborate(new Module {
  val child = Module(new Child())
  child.hardware := DontCare
  val direction = DataMirror.directionOf(child.hardware)
})
```
```scala mdoc:crash
// Not this...
elaborate(new Module {
val child = Module(new Child())
  child.hardware := DontCare
  val direction = DataMirror.directionOf(child.chiselType)
})
```

Can call `specifiedDirectionOf` on hardware or Chisel type:
```scala mdoc:silent
elaborate(new Module {
  val child = Module(new Child())
  child.hardware := DontCare
  val direction0 = DataMirror.specifiedDirectionOf(child.hardware)
  val direction1 = DataMirror.specifiedDirectionOf(child.chiselType)
})
```

## `.asInstanceOf` vs `.asTypeOf` vs `chiselTypeOf`

`.asInstanceOf` is a Scala runtime cast, usually used for telling the compiler
that you have more information than it can infer to convert Scala types:

```scala mdoc:silent
class ScalaCastingModule(gen: () => Bundle) extends Module {
  val io = IO(Output(gen().asInstanceOf[MyBundle]))
  io.foo := 0.U
}
```

This works if we do indeed have more information than the compiler:
```scala mdoc:silent
elaborate(new ScalaCastingModule( () => new MyBundle(3)))
```

But if we are wrong, we can get a Scala runtime exception:
```scala mdoc:crash
class NotMyBundle extends Bundle {val baz = Bool()}
elaborate(new ScalaCastingModule(() => new NotMyBundle()))
```

`.asTypeOf` is a conversion from one `Data` subclass to another.
It is commonly used to assign data to all-zeros, as described in [this cookbook recipe](https://www.chisel-lang.org/chisel3/docs/cookbooks/cookbook.html#how-can-i-tieoff-a-bundlevec-to-0), but it can
also be used (though not really recommended, as there is no checking on
width matches) to convert one Chisel type to another:

```scala mdoc
class SimilarToMyBundle(w: Int) extends Bundle{
  val foobar = UInt((2*w).W)
}

ChiselStage.emitSystemVerilog(new Module {
  val in = IO(Input(new MyBundle(3)))
  val out = IO(Output(new SimilarToMyBundle(3)))

  out := in.asTypeOf(out)
})
```

In contrast to `asInstanceOf` and `asTypeOf`,
`chiselTypeOf` is not a casting operation. It returns a Scala object which
can be used as shown in the examples above to create more Chisel types and
hardware with the same Chisel type as existing hardware.
