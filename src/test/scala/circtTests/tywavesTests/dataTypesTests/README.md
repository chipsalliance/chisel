# Annotations for Chisel data types + IO, Reg, and Wire bindings

For simplicity, this document reports only the outputs for IO bindings. Reg and Wire bindings have similar
representations.

## Clock, SyncReset, AsyncReset and implicit Reset (manually defined)

```scala
class TopCircuitClockReset extends RawModule {
  val clock     : Clock      = IO(Input(Clock()))
  // Types od reset https://www.chisel-lang.org/docs/explanations/reset
  val syncReset : Bool       = IO(Input(Bool()))
  val asyncReset: AsyncReset = IO(Input(AsyncReset()))
  val reset     : Reset      = IO(Input(Reset()))
}

```

```fir
FIRRTL version 4.0.0
circuit TopCircuitClockReset :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClockReset|TopCircuitClockReset",
    "typeName":"TopCircuitClockReset"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClockReset|TopCircuitClockReset>clock",
    "typeName":"IO[Clock]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClockReset|TopCircuitClockReset>syncReset",
    "typeName":"IO[Bool]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClockReset|TopCircuitClockReset>reset",
    "typeName":"IO[Reset]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClockReset|TopCircuitClockReset>asyncReset",
    "typeName":"IO[AsyncReset]"
  }
]]
  public module TopCircuitClockReset : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 89:9]
    input clock : Clock @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 90:26]
    input syncReset : UInt<1> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 92:30]
    input reset : Reset @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 93:31]
    input asyncReset : AsyncReset @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 94:36]
    skip
```

## Implicit Clock and Reset (implicitly defined)

```scala
class TopCircuitImplicitClockReset extends Module

```

```fir
FIRRTL version 4.0.0
circuit TopCircuitImplicitClockReset :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitImplicitClockReset|TopCircuitImplicitClockReset",
    "typeName":"TopCircuitImplicitClockReset"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitImplicitClockReset|TopCircuitImplicitClockReset>clock",
    "typeName":"IO[Clock]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitImplicitClockReset|TopCircuitImplicitClockReset>reset",
    "typeName":"IO[Bool]"
  }
]]
  public module TopCircuitImplicitClockReset : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 98:9]
    input clock : Clock @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 98:9]
    input reset : UInt<1> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 98:9]
    skip
```

## Ground types

```scala
class TopCircuitGroundTypes extends RawModule {
  val uint  : UInt   = IO(Output(UInt(8.W)))
  val sint  : SInt   = IO(Output(SInt(8.W)))
  val bool  : Bool   = IO(Output(Bool()))
  val analog: Analog = IO(Output(Analog(1.W)))
  // val fixedPoint: FixedPoint TODO: does fixed point still exist?
  // val interval: Interval =  IO(Output(Interval()))) TODO: does interval still exist?
  val bits  : UInt   = IO(Output(Bits(8.W)))
}

```

```fir
circuit TopCircuitGroundTypes :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitGroundTypes|TopCircuitGroundTypes",
    "typeName":"TopCircuitGroundTypes"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitGroundTypes|TopCircuitGroundTypes>uint",
    "typeName":"IO[UInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitGroundTypes|TopCircuitGroundTypes>sint",
    "typeName":"IO[SInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitGroundTypes|TopCircuitGroundTypes>bool",
    "typeName":"IO[Bool]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitGroundTypes|TopCircuitGroundTypes>analog",
    "typeName":"IO[Analog<1>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitGroundTypes|TopCircuitGroundTypes>bits",
    "typeName":"IO[UInt<8>]"
  }
]]
  public module TopCircuitGroundTypes : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 101:9]
    output uint : UInt<8> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    output sint : SInt<8> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    output bool : UInt<1> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    output analog : Analog<1> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    output bits : UInt<8> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    skip
```

## Bundles

```scala
// Test Bundles
class MyEmptyBundle extends Bundle

class MyBundle extends Bundle {
  val a: UInt = UInt(8.W)
  val b: SInt = SInt(8.W)
  val c: Bool = Bool()
}

class TopCircuitBundles extends RawModule {
  val a: Bundle        = IO(Output(new Bundle {}))
  val b: MyEmptyBundle = IO(Output(new MyEmptyBundle))
  val c: MyBundle      = IO(Output(new MyBundle))
}

```

```fir
circuit TopCircuitBundles :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundles|TopCircuitBundles",
    "typeName":"TopCircuitBundles"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundles|TopCircuitBundles>a",
    "typeName":"IO[AnonymousBundle]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundles|TopCircuitBundles>b",
    "typeName":"IO[MyEmptyBundle]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundles|TopCircuitBundles>c.c",
    "typeName":"IO[Bool]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundles|TopCircuitBundles>c.b",
    "typeName":"IO[SInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundles|TopCircuitBundles>c.a",
    "typeName":"IO[UInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundles|TopCircuitBundles>c",
    "typeName":"IO[MyBundle]"
  }
]]
  public module TopCircuitBundles : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 120:9]
    output a : { } @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    output b : { } @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    output c : { a : UInt<8>, b : SInt<8>, c : UInt<1>} @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    skip
```

## Nested Bundles

```scala
class MyNestedBundle extends Bundle {
  val a: Bool     = Bool()
  val b: MyBundle = new MyBundle
  val c: MyBundle = Flipped(new MyBundle)
}

class TopCircuitBundlesNested extends RawModule {
  val a: MyNestedBundle = IO(Output(new MyNestedBundle))
}

```

```fir
FIRRTL version 4.0.0
circuit TopCircuitBundlesNested :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested",
    "typeName":"TopCircuitBundlesNested"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a.c.c",
    "typeName":"IO[Bool]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a.c.b",
    "typeName":"IO[SInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a.c.a",
    "typeName":"IO[UInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a.c",
    "typeName":"IO[MyBundle]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a.b.c",
    "typeName":"IO[Bool]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a.b.b",
    "typeName":"IO[SInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a.b.a",
    "typeName":"IO[UInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a.b",
    "typeName":"IO[MyBundle]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a.a",
    "typeName":"IO[Bool]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundlesNested|TopCircuitBundlesNested>a",
    "typeName":"IO[MyNestedBundle]"
  }
]]
  public module TopCircuitBundlesNested : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 133:9]
    output a : { a : UInt<1>, b : { a : UInt<8>, b : SInt<8>, c : UInt<1>}, flip c : { a : UInt<8>, b : SInt<8>, c : UInt<1>}} @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    skip

```

## Vecs

For each vector, the annotation reports:

- The type of the vector
- The type of the elements of each dimension of the vector (only the first element of the dimension is reported since
  the Vec has homogeneous types)

For MixedVecs, the annotation reports the types of each of the MixedVec. This is because a MixedVec is effectively a
Bundle in which the fields are numbers.

```scala
class TopCircuitVecs(bindingChoice: BindingChoice) extends TywavesTestModule(bindingChoice) {
  val a: Vec[SInt]      = IO(Output(Vec(5, SInt(23.W))))
  val b: Vec[Vec[SInt]] = IO(Output(Vec(5, Vec(3, SInt(23.W)))))
  val c                 = IO(Output(Vec(5, new Bundle {
    val x: UInt = UInt(8.W)
  })))
  // TODO: check if this should have a better representation, now its type is represented as other Records
  val d                 = IO(Output(MixedVec(UInt(3.W), SInt(10.W))))
}

```

```fir
FIRRTL version 4.0.0
circuit TopCircuitVecs :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs",
    "typeName":"TopCircuitVecs"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>a[0]",
    "typeName":"IO[SInt<23>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>a",
    "typeName":"IO[SInt<23>[5]]",
    "params":[
      {
        "name":"length",
        "typeName":"Int"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>b[0][0]",
    "typeName":"IO[SInt<23>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>b[0]",
    "typeName":"IO[SInt<23>[3]]",
    "params":[
      {
        "name":"length",
        "typeName":"Int"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>b",
    "typeName":"IO[SInt<23>[3][5]]",
    "params":[
      {
        "name":"length",
        "typeName":"Int"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>c[0].x",
    "typeName":"IO[UInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>c[0]",
    "typeName":"IO[AnonymousBundle]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>c",
    "typeName":"IO[AnonymousBundle[5]]",
    "params":[
      {
        "name":"length",
        "typeName":"Int"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>d.0",
    "typeName":"IO[UInt<3>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>d.1",
    "typeName":"IO[SInt<10>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitVecs|TopCircuitVecs>d",
    "typeName":"IO[MixedVec]"
  }
]]
  public module TopCircuitVecs : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 195:11]
    output a : SInt<23>[5] @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 43:51]
    output b : SInt<23>[3][5] @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 43:51]
    output c : { x : UInt<8>}[5] @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 43:51]
    output d : { `1` : SInt<10>, `0` : UInt<3>} @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 43:51]

    skip
```

## Vecs as fields in bundles

```scala
// Test Bundles with Vecs
class TopCircuitBundleWithVec extends RawModule {
  val a = IO(Output(new Bundle {
    val vec = Vec(5, UInt(8.W))
  }))
}

```

```fir
circuit TopCircuitBundleWithVec :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundleWithVec|TopCircuitBundleWithVec",
    "typeName":"TopCircuitBundleWithVec"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundleWithVec|TopCircuitBundleWithVec>a.vec[0]",
    "typeName":"IO[UInt<8>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundleWithVec|TopCircuitBundleWithVec>a.vec",
    "typeName":"IO[UInt<8>[5]]",
    "params":[
      {
        "name":"length",
        "typeName":"Int"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBundleWithVec|TopCircuitBundleWithVec>a",
    "typeName":"IO[AnonymousBundle]"
  }
]]
  public module TopCircuitBundleWithVec : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 149:9]
    output a : { vec : UInt<8>[5]} @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 13:51]
    skip
```