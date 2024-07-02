# Annotations for Chisel Modules

For each circuit in [ModuleCircuits](../TywavesAnnotationCircuits.scala), the following annotations are generated:

## Empty Circuit

This test checks that the annotation is generated for a circuit.

```scala
class TopCircuit extends RawModule

```

```fir
circuit TopCircuit :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuit|TopCircuit",
    "typeName":"TopCircuit"
  }]]
public module TopCircuit: @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 37: 9]
skip
```

```verilog
module TopCircuit();	// src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala:75:11
endmodule
```

## Circuit with a submodule

This test checks the annotation for a submodule in a circuit.

```scala
class MyModule extends RawModule

class TopCircuitSubModule extends RawModule {
  val mod = Module(new MyModule)
}

```

```fir
circuit TopCircuitSubModule :%[[
    {
        "class": "chisel3.tywaves.TywavesAnnotation",
        "target": "~TopCircuitSubModule|MyModule",
        "typeName": "MyModule"
    },
    {
        "class": "chisel3.tywaves.TywavesAnnotation",
        "target": "~TopCircuitSubModule|TopCircuitSubModule",
        "typeName": "TopCircuitSubModule"
    }]]
    module MyModule: @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 40: 9]
        skip
    public module TopCircuitSubModule: @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 41: 9]
        inst mod of MyModule @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 41: 65]
```

In the final verilog this is optimized even with `"-g"` since the submodule is empty.

```verilog
module TopCircuitSubModule();	// src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala:80:11
endmodule
```

## Circuit with multiple submodules

This test checks that multiple instances of the same module are annotated.

```scala
class TopCircuitMultiModule extends RawModule {
  // 4 times MyModule in total
  val mod1 = Module(new MyModule)
  val mod2 = Module(new MyModule)
  val mods = Seq.fill(2)(Module(new MyModule))
}

```

```fir
circuit TopCircuitMultiModule :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitMultiModule|MyModule",
    "typeName":"MyModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitMultiModule|MyModule_1",
    "typeName":"MyModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitMultiModule|MyModule_2",
    "typeName":"MyModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitMultiModule|MyModule_3",
    "typeName":"MyModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitMultiModule|TopCircuitMultiModule",
    "typeName":"TopCircuitMultiModule"
  }
]]
  module MyModule : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 40:9]
    skip
  module MyModule_1 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 40:9]
    skip
  module MyModule_2 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 40:9]
    skip
  module MyModule_3 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 40:9]
    skip
    
  public module TopCircuitMultiModule : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 44:9]
    inst mod1 of MyModule @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 46:22]
    inst mod2 of MyModule_1 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 47:22]
    inst mods_0 of MyModule_2 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 48:34]
    inst mods_1 of MyModule_3 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 48:34]
```

## A circuit with parameters

Here we check that the annotation is generated for a circuit with a module that has parameters.

**Note**: at the moment it doesn't distinguish between different instances of the same module with different parameters.
The goal is to have:

- `MyModule(n: 8)`
- `MyModule(n: 16)`
- `MyModule(n: 32)`

Or something similar.

```scala
class TopCircuitWithParams extends RawModule {
  class MyModule(val width: Int) extends RawModule

  val mod1: MyModule = Module(new MyModule(8))
  val mod2: MyModule = Module(new MyModule(16))
  val mod3: MyModule = Module(new MyModule(32))
}

```

```fir
circuit TopCircuitWithParams :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitWithParams|MyModule",
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
    "target":"~TopCircuitWithParams|MyModule_1",
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
    "target":"~TopCircuitWithParams|MyModule_2",
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
    "target":"~TopCircuitWithParams|TopCircuitWithParams",
    "typeName":"TopCircuitWithParams"
  }
]]
  module MyModule : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 53:11]
    skip
  module MyModule_1 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 53:11]
    skip
  module MyModule_2 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 53:11]
    skip

  public module TopCircuitWithParams : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 52:9]
    inst mod1 of MyModule @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 54:32]
    inst mod2 of MyModule_1 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 55:32]
    inst mod3 of MyModule_2 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 56:32]
```

## Blackbox module

Blackboxes are like modules, but they have in addition parameters and ports that are not defined in the Chisel code.
These parameters are already included in the FIRRTL syntax for blackboxes.

```scala
class TopCircuitBlackBox extends RawModule {
  class MyBlackBox extends BlackBox(Map("PARAM1" -> "TRUE", "PARAM2" -> "DEFAULT")) {
    val io = IO(new Bundle {})
  }

  val myBlackBox1 : MyBlackBox      = Module(new MyBlackBox)
  val myBlackBox2 : MyBlackBox      = Module(new MyBlackBox)
  val myBlackBoxes: Seq[MyBlackBox] = Seq.fill(2)(Module(new MyBlackBox))
}

```

```fir
circuit TopCircuitBlackBox :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBlackBox|MyBlackBox",
    "typeName":"MyBlackBox"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBlackBox|MyBlackBox_1",
    "typeName":"MyBlackBox"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBlackBox|MyBlackBox_2",
    "typeName":"MyBlackBox"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBlackBox|MyBlackBox_3",
    "typeName":"MyBlackBox"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitBlackBox|TopCircuitBlackBox",
    "typeName":"TopCircuitBlackBox"
  }
]]
  extmodule MyBlackBox : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 102:13]
    defname = MyBlackBox
    parameter PARAM1 = "TRUE"
    parameter PARAM2 = "DEFAULT"
  extmodule MyBlackBox_1 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 102:13]
    defname = MyBlackBox
    parameter PARAM1 = "TRUE"
    parameter PARAM2 = "DEFAULT"
  extmodule MyBlackBox_2 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 102:13]
    defname = MyBlackBox
    parameter PARAM1 = "TRUE"
    parameter PARAM2 = "DEFAULT"
  extmodule MyBlackBox_3 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 102:13]
    defname = MyBlackBox
    parameter PARAM1 = "TRUE"
    parameter PARAM2 = "DEFAULT"

  public module TopCircuitBlackBox : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 101:11]
    inst myBlackBox1 of MyBlackBox @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 106:44]
    inst myBlackBox2 of MyBlackBox_1 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 107:44]
    inst myBlackBoxes_0 of MyBlackBox_2 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 108:61]
    inst myBlackBoxes_1 of MyBlackBox_3 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 108:61]
```

## Intrinsic module

Similar to blackboxes, also intrinsic modules have parameters which are already "defined" in the FIRRTL syntax.

```scala
class ExampleIntrinsicModule(str: String) extends IntrinsicModule("OtherIntrinsic", Map("STRING" -> str)) {}

class TopCircuitIntrinsic extends RawModule {
  val myIntrinsicModule1: ExampleIntrinsicModule      = Module(new ExampleIntrinsicModule("Hello"))
  val myIntrinsicModule2: ExampleIntrinsicModule      = Module(new ExampleIntrinsicModule("World"))
  val myIntrinsicModules: Seq[ExampleIntrinsicModule] = Seq.fill(2)(Module(new ExampleIntrinsicModule("Hello")))
}

```

```fir
circuit TopCircuitIntrinsic :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitIntrinsic|ExampleIntrinsicModule",
    "typeName":"ExampleIntrinsicModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitIntrinsic|ExampleIntrinsicModule_1",
    "typeName":"ExampleIntrinsicModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitIntrinsic|ExampleIntrinsicModule_2",
    "typeName":"ExampleIntrinsicModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitIntrinsic|ExampleIntrinsicModule_3",
    "typeName":"ExampleIntrinsicModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitIntrinsic|TopCircuitIntrinsic",
    "typeName":"TopCircuitIntrinsic"
  }
]]
  intmodule ExampleIntrinsicModule : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationSpec.scala 137:13]
    intrinsic = OtherIntrinsic
    parameter STRING = "Hello"
  intmodule ExampleIntrinsicModule_1 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationSpec.scala 137:13]
    intrinsic = OtherIntrinsic
    parameter STRING = "World"
  intmodule ExampleIntrinsicModule_2 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationSpec.scala 137:13]
    intrinsic = OtherIntrinsic
    parameter STRING = "Hello"
  intmodule ExampleIntrinsicModule_3 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationSpec.scala 137:13]
    intrinsic = OtherIntrinsic
    parameter STRING = "Hello"
  public module TopCircuitIntrinsic : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationSpec.scala 138:13]

    inst myIntrinsicModule1 of ExampleIntrinsicModule @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationSpec.scala 139:64]
    inst myIntrinsicModule2 of ExampleIntrinsicModule_1 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationSpec.scala 140:64]
    inst myIntrinsicModules_0 of ExampleIntrinsicModule_2 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationSpec.scala 141:81]
    inst myIntrinsicModules_1 of ExampleIntrinsicModule_3 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationSpec.scala 141:81]
```

## Classes

```scala
// An abstract description of a CSR, represented as a Class.
@instantiable
class CSRDescription extends Class

class CSRModule(csrDescDef: Definition[CSRDescription]) extends RawModule {
  val csrDescription = Instance(csrDescDef)
}

class TopCircuitClasses extends RawModule {
  val csrDescDef                 = Definition(new CSRDescription)
  val csrModule1: CSRModule      = Module(new CSRModule(csrDescDef))
  val csrModule2: CSRModule      = Module(new CSRModule(csrDescDef))
  val csrModules: Seq[CSRModule] = Seq.fill(2)(Module(new CSRModule(csrDescDef)))
}

```

```fir
circuit TopCircuitClasses :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClasses|CSRDescription",
    "typeName":"CSRDescription"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClasses|CSRModule",
    "typeName":"CSRModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClasses|CSRModule_1",
    "typeName":"CSRModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClasses|CSRModule_2",
    "typeName":"CSRModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClasses|CSRModule_3",
    "typeName":"CSRModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitClasses|TopCircuitClasses",
    "typeName":"TopCircuitClasses"
  }
]]
  class CSRDescription : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 76:4]
    skip
    
  module CSRModule : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 78:9]
    object csrDescription of CSRDescription @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 79:34]
  module CSRModule_1 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 78:9]
    object csrDescription of CSRDescription @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 79:34]
  module CSRModule_2 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 78:9]
    object csrDescription of CSRDescription @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 79:34]
  module CSRModule_3 : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 78:9]
    object csrDescription of CSRDescription @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 79:34]

  public module TopCircuitClasses : @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 81:9]
    inst csrModule1 of CSRModule @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 83:39]
    inst csrModule2 of CSRModule_1 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 84:39]
    inst csrModules_0 of CSRModule_2 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 85:56]
    inst csrModules_1 of CSRModule_3 @[src/test/scala/circtTests/tywavesTests/TywavesAnnotationCircuits.scala 85:56]
```

### Parametric modules

Here, similarly to modules with parameter, we want to be able to retrieve the parametric values in order to distinguish
different parametric types of modules.
Thus, the following example should emit:

- `MyModule[UInt(8.W)]` for `myModule1`
- `MyModule[SInt(8.W)]` for `myModule2`
- `MyModule[Bool]` for `myModule3`

```scala
class MyModule[T <: Data](gen: T) extends RawModule

class TopCircuitParametric extends RawModule {
  val myModule1: MyModule[UInt] = Module(new MyModule(UInt(8.W)))
  val myModule2: MyModule[SInt] = Module(new MyModule(SInt(8.W)))
  val myModule3: MyModule[Bool] = Module(new MyModule(Bool()))
}

```

```fir
circuit TopCircuitParametric :%[[
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitParametric|MyModule",
    "typeName":"MyModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitParametric|MyModule_1",
    "typeName":"MyModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitParametric|MyModule_2",
    "typeName":"MyModule"
  },
  {
    "class":"chisel3.tywaves.TywavesAnnotation",
    "target":"~TopCircuitParametric|TopCircuitParametric",
    "typeName":"TopCircuitParametric"
  }
]]
  module MyModule : @[src/test/scala/circtTests/tywavesTests/moduleTests/TypeAnnotationModulesSpec.scala 86:13]
    skip
  module MyModule_1 : @[src/test/scala/circtTests/tywavesTests/moduleTests/TypeAnnotationModulesSpec.scala 86:13]
    skip
  module MyModule_2 : @[src/test/scala/circtTests/tywavesTests/moduleTests/TypeAnnotationModulesSpec.scala 86:13]
    skip

  public module TopCircuitParametric : @[src/test/scala/circtTests/tywavesTests/moduleTests/TypeAnnotationModulesSpec.scala 88:13]
    inst myModule1 of MyModule @[src/test/scala/circtTests/tywavesTests/moduleTests/TypeAnnotationModulesSpec.scala 89:47]
    inst myModule2 of MyModule_1 @[src/test/scala/circtTests/tywavesTests/moduleTests/TypeAnnotationModulesSpec.scala 90:47]
    inst myModule3 of MyModule_2 @[src/test/scala/circtTests/tywavesTests/moduleTests/TypeAnnotationModulesSpec.scala 91:47]
```
