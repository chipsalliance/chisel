# Annotations for Chisel Constructs with Parameters

For each circuit in [ParamCircuits](../TywavesAnnotationCircuits.scala), the following annotations are generated:

## Circuit with Parameters

> **NOTE**: `val` inside the body of the class are not considered parameters

```scala
class TopCircuitWithParams(val width1: Int, width2: Int) extends RawModule {
  val width3     = width1 + width2 // Not a parameter
  val uint: UInt = IO(Input(UInt(width1.W)))
  val sint: SInt = IO(Input(SInt(width2.W)))
  val bool: Bool = IO(Input(Bool()))
  val bits: Bits = IO(Input(Bits(width3.W)))
}

```

```fir
circuit TopCircuitWithParams :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParams|TopCircuitWithParams",
    "typeName":"TopCircuitWithParams",
    "params":[
      {
        "name":"width1",
        "typeName":"Int",
        "value":"8"
      },
      {
        "name":"width2",
        "typeName":"Int",
        "value":"16"
      }
    ]
  },
  ; Other tywaves annotations
]]
  public module TopCircuitWithParams : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 318:11]
    input uint : UInt<8> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 320:26]
    input sint : SInt<16> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 321:26]
    input bool : UInt<1> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 322:26]
    input bits : UInt<24> @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 323:26]
    skip
```

## Module instances with basic scala types parameters

```scala
class TopCircuitWithParamModules extends RawModule {
  class MyModule(val width: Int) extends RawModule

  val mod1: MyModule = Module(new MyModule(8))
  val mod2: MyModule = Module(new MyModule(16))
  val mod3: MyModule = Module(new MyModule(32))
}

```

```fir
circuit TopCircuitWithParamModules :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamModules|MyModule",
    "typeName":"MyModule",
    "params":[
      {
        "name":"width",
        "typeName":"Int",
        "value":"8"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamModules|MyModule_1",
    "typeName":"MyModule",
    "params":[
      {
        "name":"width",
        "typeName":"Int",
        "value":"16"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamModules|MyModule_2",
    "typeName":"MyModule",
    "params":[
      {
        "name":"width",
        "typeName":"Int",
        "value":"32"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamModules|TopCircuitWithParamModules",
    "typeName":"TopCircuitWithParamModules"
  }
]]
  module MyModule : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 328:13]
    skip
  module MyModule_1 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 328:13]
    skip
  module MyModule_2 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 328:13]
    skip

  public module TopCircuitWithParamModules : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 327:11]
    inst mod1 of MyModule @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 330:34]
    inst mod2 of MyModule_1 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 331:34]
    inst mod3 of MyModule_2 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 332:34]
```

## Bundles with chisel types parameters

```scala
class TopCircuitWithParamBundle extends RawModule {
  class BaseBundle(val n: Int) extends Bundle {
    val b: UInt = UInt(n.W)
  }

  class OtherBundle(val a: UInt, val b: BaseBundle) extends Bundle {} // Example of nested class in parameters

  class TopBundle(a: Bool, val b: String, protected val c: Char, private val d: Boolean, val o: OtherBundle)
    extends Bundle {
    val inner_a = a
  }

  case class CaseClassExample(a: Int, o: OtherBundle) extends Bundle

  val baseBundle  = IO(Input(new BaseBundle(1)))
  val otherBundle = IO(Input(new OtherBundle(UInt(baseBundle.n.W), baseBundle.cloneType)))
  val topBundle   = IO(Input(new TopBundle(Bool(), "hello", 'c', true, otherBundle.cloneType)))

  val caseClassBundle = IO(Input(CaseClassExample(1, new OtherBundle(UInt(2.W), baseBundle.cloneType))))

}

```

```fir
circuit TopCircuitWithParamBundle :%[[
  ; other tywaves annotations
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>baseBundle",
    "typeName":"IO[BaseBundle]",
    "params":[
      {
        "name":"n",
        "typeName":"Int",
        "value":"1"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>otherBundle.b",
    "typeName":"IO[BaseBundle]",
    "params":[
      {
        "name":"n",
        "typeName":"Int",
        "value":"1"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>otherBundle",
    "typeName":"IO[OtherBundle]",
    "params":[
      {
        "name":"a",
        "typeName":"UInt",
        "value":"IO[UInt<1>]"
      },
      {
        "name":"b",
        "typeName":"BaseBundle",
        "value":"BaseBundle(n: 1)"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>topBundle.o.b.b",
    "typeName":"IO[UInt<1>]"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>topBundle.o.b",
    "typeName":"IO[BaseBundle]",
    "params":[
      {
        "name":"n",
        "typeName":"Int",
        "value":"1"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>topBundle.o",
    "typeName":"IO[OtherBundle]",
    "params":[
      {
        "name":"a",
        "typeName":"UInt",
        "value":"IO[UInt<1>]"
      },
      {
        "name":"b",
        "typeName":"BaseBundle",
        "value":"BaseBundle(n: 1)"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>topBundle",
    "typeName":"IO[TopBundle]",
    "params":[
      {
        "name":"a",
        "typeName":"Bool",
        "value":"IO[Bool]"
      },
      {
        "name":"b",
        "typeName":"String",
        "value":"hello"
      },
      {
        "name":"c",
        "typeName":"Char",
        "value":"c"
      },
      {
        "name":"d",
        "typeName":"Boolean",
        "value":"true"
      },
      {
        "name":"o",
        "typeName":"OtherBundle",
        "value":"OtherBundle(a: IO[UInt<1>], b: BaseBundle(n: 1))"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>caseClassBundle.o.b",
    "typeName":"IO[BaseBundle]",
    "params":[
      {
        "name":"n",
        "typeName":"Int",
        "value":"1"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>caseClassBundle.o",
    "typeName":"IO[OtherBundle]",
    "params":[
      {
        "name":"a",
        "typeName":"UInt",
        "value":"IO[UInt<2>]"
      },
      {
        "name":"b",
        "typeName":"BaseBundle",
        "value":"BaseBundle(n: 1)"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamBundle|TopCircuitWithParamBundle>caseClassBundle",
    "typeName":"IO[CaseClassExample]",
    "params":[
      {
        "name":"a",
        "typeName":"Int",
        "value":"1"
      },
      {
        "name":"o",
        "typeName":"OtherBundle",
        "value":"OtherBundle(a: IO[UInt<2>], b: BaseBundle(n: 1))"
      }
    ]
  }
]]
  public module TopCircuitWithParamBundle : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 336:11]
    input baseBundle : { b : UInt<1>} @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 348:26]
    input otherBundle : { a : UInt<1>, b : { b : UInt<1>}} @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 349:27]
    input topBundle : { inner_a : UInt<1>, o : { a : UInt<1>, b : { b : UInt<1>}}} @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 350:25]
    input caseClassBundle : { o : { a : UInt<2>, b : { b : UInt<1>}}} @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 352:31]

    skip


```

## Bundles and modules with scala classes and case classes types parameters

> **NOTE**: when the parameter is not a `val` is not always annotated

```scala
class TopCircuitWithParamScalaClasses extends RawModule {
  class MyScalaClass(val a: Int, b: String)

  case class MyScalaCaseClass(a: Int, b: String)

  class MyModule(val a: MyScalaClass, b: MyScalaCaseClass) extends RawModule

  class MyBundle(val a: MyScalaClass, b: MyScalaCaseClass) extends Bundle

  val mod   : MyModule = Module(new MyModule(new MyScalaClass(1, "hello"), MyScalaCaseClass(2, "world")))
  val bundle: MyBundle = IO(Input(new MyBundle(new MyScalaClass(1, "hello"), MyScalaCaseClass(2, "world"))))

}

```

```fir
circuit TopCircuitWithParamScalaClasses :%[[
  ; other tywaves annotations
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamScalaClasses|MyModule",
    "typeName":"MyModule",
    "params":[
      {
        "name":"a",
        "typeName":"MyScalaClass",
        "value":"MyScalaClass(a: 1, b)"
      },
      {
        "name":"b",
        "typeName":"MyScalaCaseClass",
        "value":"MyScalaCaseClass(a: 2, b: world)"
      }
    ]
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParamScalaClasses|TopCircuitWithParamScalaClasses>bundle",
    "typeName":"IO[MyBundle]",
    "params":[
      {
        "name":"a",
        "typeName":"MyScalaClass",
        "value":"MyScalaClass(a: 1, b)"
      },
      {
        "name":"b",
        "typeName":"MyScalaCaseClass",
        "value":"MyScalaCaseClass(a: 2, b: world)"
      }
    ]
  }
]]
  module MyModule : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 361:13]
    skip

  public module TopCircuitWithParamScalaClasses : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 357:11]
    input bundle : { } @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 365:32]
    inst mod of MyModule @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 364:36]

```
