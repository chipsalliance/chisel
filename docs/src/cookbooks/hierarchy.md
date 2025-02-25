---
sidebar_position: 2
---

# Hierarchy Cookbook

import TOCInline from '@theme/TOCInline';

<TOCInline toc={toc} />

## How do I instantiate multiple instances with the same module parameterization?

Prior to this package, Chisel users relied on deduplication in a FIRRTL compiler to combine
structurally equivalent modules into one module (aka "deduplication").
This package introduces the following new APIs to enable multiply-instantiated modules directly in Chisel.

`Definition(...)` enables elaborating a module, but does not actually instantiate that module.
Instead, it returns a `Definition` class which represents that module's definition.

`Instance(...)` takes a `Definition` and instantiates it, returning an `Instance` object.

`Instantiate(...)` provides an API similar to `Module(...)`, except it uses
`Definition` and `Instance` to only elaborate modules once for a given set of
parameters. It returns an `Instance` object.

Modules (classes or traits) which will be used with the `Definition`/`Instance` api should be marked
with the `@instantiable` annotation at the class/trait definition.

To make a Module's members variables accessible from an `Instance` object, they must be annotated
with the `@public` annotation. Note that this is only accessible from a Scala sense—this is not
in and of itself a mechanism for cross-module references.

### Using Definition and Instance

In the following example, use `Definition`, `Instance`, `@instantiable` and `@public` to create
multiple instances of one specific parameterization of a module, `AddOne`.

```scala mdoc:silent
import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}

@instantiable
class AddOne(width: Int) extends Module {
  @public val in  = IO(Input(UInt(width.W)))
  @public val out = IO(Output(UInt(width.W)))
  out := in + 1.U
}

class AddTwo(width: Int) extends Module {
  val in  = IO(Input(UInt(width.W)))
  val out = IO(Output(UInt(width.W)))
  val addOneDef = Definition(new AddOne(width))
  val i0 = Instance(addOneDef)
  val i1 = Instance(addOneDef)
  i0.in := in
  i1.in := i0.out
  out   := i1.out
}
```
```scala mdoc:verilog
chisel3.docs.emitSystemVerilog(new AddTwo(10))
```

### Using Instantiate

Similar to the above, the following example uses `Instantiate` to create
multiple instances of `AddOne`.

```scala mdoc:silent
import chisel3.experimental.hierarchy.Instantiate

class AddTwoInstantiate(width: Int) extends Module {
  val in  = IO(Input(UInt(width.W)))
  val out = IO(Output(UInt(width.W)))
  val i0 = Instantiate(new AddOne(width))
  val i1 = Instantiate(new AddOne(width))
  i0.in := in
  i1.in := i0.out
  out   := i1.out
}
```
```scala mdoc:verilog
chisel3.docs.emitSystemVerilog(new AddTwoInstantiate(16))
```

## How do I access internal fields of an instance?

You can mark internal members of a Module class or trait marked with `@instantiable` with the `@public` annotation.
The requirements are that the field is publicly accessible, is a `val` or `lazy val`, and must have an implementation of `Lookupable`.

Types that are supported by default are:

1. `Data`
2. `BaseModule`
3. `MemBase`
4. `IsLookupable`
5. `Iterable`/`Option`/`Either` containing a type that meets these requirements
6. Basic type like `String`, `Int`, `BigInt`, `Unit`, etc.

To mark a superclass's member as `@public`, use the following pattern (shown with `val clock`).

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class MyModule extends Module {
  @public val clock = clock
}
```

You'll get the following error message for improperly marking something as `@public`:

```scala mdoc:reset:fail
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}

object NotValidType

@instantiable
class MyModule extends Module {
  @public val x = NotValidType
}
```

## How do I make my fields accessible from an instance?

If an instance's fields are simple (e.g. `Int`, `String` etc.) they can be marked directly with `@public`.

Often, fields are more complicated (e.g. a user-defined case class).
If a case class is only made up of simple types (i.e. it does *not* contain any `Data`, `BaseModules`, memories, or `Instances`),
it can extend the `IsLookupable` trait.
This indicates to Chisel that instances of the `IsLookupable` class may be accessed from within instances.
(If the class *does* contain things like `Data` or modules, [the section below](#how-do-i-make-case-classes-containing-data-or-modules-accessible-from-an-instance).)

However, ensure that these parameters are true for **all** instances of a definition.
For example, if our parameters contained an id field which was instance-specific but defaulted to zero,
then the definition's id would be returned for all instances.
This change in behavior could lead to bugs if other code presumed the id field was correct.

Thus, it is important that when converting normal modules to use this package,
you are careful about what you mark as `IsLookupable`.

In the following example, we added the trait `IsLookupable` to allow the member to be marked `@public`.

```scala mdoc:reset:silent
import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, IsLookupable, public}

case class MyCaseClass(width: Int) extends IsLookupable

@instantiable
class MyModule extends Module {
  @public val x = MyCaseClass(10)
}

class Top extends Module {
  val inst = Instance(Definition(new MyModule))
  println(s"Width is ${inst.x.width}")
}
```
```scala mdoc:passthrough
println("```")
// Run elaboration so that the println above shows up
circt.stage.ChiselStage.elaborate(new Top)
println("```")
```

## How do I make case classes containing Data or Modules accessible from an instance?

For case classes containing `Data`, `BaseModule`, `MemBase` or `Instance` types, you can provide an implementation of the `Lookupable` typeclass.

**Note that Lookupable for Modules is deprecated, please cast to Instance instead (with `.toInstance`).**

Consider the following case class:

```scala mdoc:reset
import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}

@instantiable
class MyModule extends Module {
  @public val wire = Wire(UInt(8.W))
}
case class UserDefinedType(name: String, data: UInt, inst: Instance[MyModule])
```

By default, instances of `UserDefinedType` will not be accessible from instances:

```scala mdoc:fail
@instantiable
class HasUserDefinedType extends Module {
  val inst = Module(new MyModule)
  val wire = Wire(UInt(8.W))
  @public val x = UserDefinedType("foo", wire, inst.toInstance)
}
```

We can implement the `Lookupable` type class for `UserDefinedType` in order to make it accessible.
This involves defining an implicit val in the companion object for `UserDefinedType`.
Because `UserDefinedType` has three fields, we use the `Lookupable.product3` factory.
It takes 4 type parameters: the type of the case class, and the types of each of its fields.

**If any fields are `BaseModules`, you must change them to be `Instance[_]` in order to define the `Lookupable` typeclass.**

For more information about typeclasses, see the [DataView section on Type Classes](https://www.chisel-lang.org/chisel3/docs/explanations/dataview#type-classes).

```scala mdoc
import chisel3.experimental.hierarchy.Lookupable
object UserDefinedType {
  // Use Lookupable.Simple type alias as return type.
  implicit val lookupable: Lookupable.Simple[UserDefinedType] =
    Lookupable.product3[UserDefinedType, String, UInt, Instance[MyModule]](
      // Provide the recipe for converting the UserDefinedType to a Tuple.
      x => (x.name, x.data, x.inst),
      // Provide the recipe for converting a Tuple to a user defined type.
      // For case classes, you can use the built-in factory method.
      UserDefinedType.apply
    )
}
```

Now, we can access instances of `UserDefinedType` from instances:

```scala mdoc
@instantiable
class HasUserDefinedType extends Module {
  val inst = Module(new MyModule)
  val wire = Wire(UInt(8.W))
  @public val x = UserDefinedType("foo", wire, inst.toInstance)
}
class Top extends Module {
  val inst = Instance(Definition(new HasUserDefinedType))
  println(s"Name is: ${inst.x.name}")
}
```

## How do I make type parameterized case classes accessible from an instance?

Consider the following type-parameterized case class:

```scala mdoc:reset
import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}

case class ParameterizedUserDefinedType[A, T <: Data](value: A, data: T)
```

Similarly to `HasUserDefinedType` we need to define an implicit to provide the `Lookupable` typeclass.
Unlike the simpler example above, however, we use an `implicit def` to handle the type parameters:

```scala mdoc
import chisel3.experimental.hierarchy.Lookupable
object ParameterizedUserDefinedType {
  // Type class materialization is recursive, so both A and T must have Lookupable instances.
  // We required this for A via the context bound `: Lookupable`.
  // Data is a Chisel built-in so is known to have a Lookupable instance.
  implicit def lookupable[A : Lookupable, T <: Data]: Lookupable.Simple[ParameterizedUserDefinedType[A, T]] =
    Lookupable.product2[ParameterizedUserDefinedType[A, T], A, T](
      x => (x.value, x.data),
      ParameterizedUserDefinedType.apply
    )
}
```

Now, we can access instances of `ParameterizedUserDefinedType` from instances:

```scala mdoc
class ChildModule extends Module {
  @public val wire = Wire(UInt(8.W))
}
@instantiable
class HasUserDefinedType extends Module {
  val wire = Wire(UInt(8.W))
  @public val x = ParameterizedUserDefinedType("foo", wire)
  @public val y = ParameterizedUserDefinedType(List(1, 2, 3), wire)
}
class Top extends Module {
  val inst = Instance(Definition(new HasUserDefinedType))
  println(s"x.value is: ${inst.x.value}")
  println(s"y.value.head is: ${inst.y.value.head}")
}
```

## How do I make case classes with lots of fields accessible from an instance?

Lookupable provides factories for `product1` to `product5`.
If your class has more than 5 fields, you can use nested tuples as "pseduo-fields" in the mapping.

```scala mdoc
case class LotsOfFields(a: Data, b: Data, c: Data, d: Data, e: Data, f: Data)
object LotsOfFields {
  implicit val lookupable: Lookupable.Simple[LotsOfFields] =
    Lookupable.product5[LotsOfFields, Data, Data, Data, Data, (Data, Data)](
      x => (x.a, x.b, x.c, x.d, (x.e, x.f)),
      // Cannot use factory method directly this time since we have to unpack the tuple.
      { case (a, b, c, d, (e, f)) => LotsOfFields(a, b, c, d, e, f) },
    )
}
```

## How do I look up fields from a Definition, if I don't want to instantiate it?

Just like `Instance`s, `Definition`'s also contain accessors for `@public` members.
As such, you can directly access them:

```scala mdoc:reset:silent
import chisel3._
import chisel3.experimental.hierarchy.{Definition, instantiable, public}

@instantiable
class AddOne(val width: Int) extends RawModule {
  @public val width = width
  @public val in  = IO(Input(UInt(width.W)))
  @public val out = IO(Output(UInt(width.W)))
  out := in + 1.U
}

class Top extends Module {
  val definition = Definition(new AddOne(10))
  println(s"Width is: ${definition.width}")
}
```

```scala mdoc:verilog
chisel3.docs.emitSystemVerilog(new Top())
```

## How do I parameterize a module by its children instances?

Prior to the introduction of this package, a parent module would have to pass all necessary parameters
when instantiating a child module.
This had the unfortunate consequence of requiring a parent's parameters to always contain the child's
parameters, which was an unnecessary coupling which lead to some anti-patterns.

Now, a parent can take a child `Definition` as an argument, and instantiate it directly.
In addition, it can analyze the parameters used in the definition to parameterize itself.
In a sense, now the child can actually parameterize the parent.

In the following example, we create a definition of `AddOne`, and pass the definition to `AddTwo`.
The width of the `AddTwo` ports are now derived from the parameterization of the `AddOne` instance.

```scala mdoc:reset
import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance, instantiable, public}

@instantiable
class AddOne(val width: Int) extends Module {
  @public val width = width
  @public val in  = IO(Input(UInt(width.W)))
  @public val out = IO(Output(UInt(width.W)))
  out := in + 1.U
}

class AddTwo(addOneDef: => Definition[AddOne]) extends Module {
  private val definition = addOneDef
  val i0 = Instance(definition)
  val i1 = Instance(definition)
  val in  = IO(Input(UInt(definition.width.W)))
  val out = IO(Output(UInt(definition.width.W)))
  i0.in := in
  i1.in := i0.out
  out   := i1.out
}
```
```scala mdoc:verilog
chisel3.docs.emitSystemVerilog(new AddTwo(Definition(new AddOne(10))))
```

## How do I use the new hierarchy-specific Select functions?

Select functions can be applied after a module has been elaborated, either in a Chisel Aspect or in a parent module applied to a child module.

There are seven hierarchy-specific functions, which (with the exception of `ios`) either return `Instance`'s or `Definition`'s:
 - `instancesIn(parent)`: Return all instances directly instantiated locally within `parent`
 - `instancesOf[type](parent)`: Return all instances of provided `type` directly instantiated locally within `parent`
 - `allInstancesOf[type](root)`: Return all instances of provided `type` directly and indirectly instantiated, locally and deeply, starting from `root`
 - `definitionsIn`: Return definitions of all instances directly instantiated locally within `parent`
 - `definitionsOf[type]`: Return definitions of all instances of provided `type` directly instantiated locally within `parent`
 - `allDefinitionsOf[type]`: Return all definitions of instances of provided `type` directly and indirectly instantiated, locally and deeply, starting from `root`
 - `ios`: Returns all the I/Os of the provided definition or instance.

To demonstrate this, consider the following. We mock up an example where we are using the `Select.allInstancesOf` and `Select.allDefinitionsOf` to annotate instances and the definition of `EmptyModule`.
When the annotation logic is execute after elaboration, we print the resulting `Target`.
As shown, despite `EmptyModule` actually only being elaborated once, we still provide different targets depending on how the instance or definition is selected.

```scala mdoc:reset
import chisel3._
import chisel3.experimental.hierarchy.{Definition, Instance, Hierarchy, instantiable, public}

@instantiable
class EmptyModule extends Module {
  println("Elaborating EmptyModule!")
}

@instantiable
class TwoEmptyModules extends Module {
  val definition = Definition(new EmptyModule)
  val i0         = Instance(definition)
  val i1         = Instance(definition)
}

class Top extends Module {
  val definition = Definition(new TwoEmptyModules)
  val instance   = Instance(definition)
  aop.Select.allInstancesOf[EmptyModule](instance).foreach { i =>
    experimental.annotate(i) {
      println("instance: " + i.toTarget)
      Nil
    }
  }
  aop.Select.allDefinitionsOf[EmptyModule](instance).foreach { d =>
    experimental.annotate(d) {
      println("definition: " + d.toTarget)
      Nil
    }
  }
}
```
```scala mdoc:passthrough
println("```")
val x = circt.stage.ChiselStage.emitCHIRRTL(new Top)
println("```")
```

You can also use `Select.ios` on either a `Definition` or an `Instance` to annotate the I/Os appropriately:

```scala mdoc
@instantiable
class InOutModule extends Module {
  @public val in = IO(Input(Bool()))
  @public val out = IO(Output(Bool()))
  out := in
}

@instantiable
class TwoInOutModules extends Module {
  val in = IO(Input(Bool()))
  val out = IO(Output(Bool()))
  val definition = Definition(new InOutModule)
  val i0         = Instance(definition)
  val i1         = Instance(definition)
  i0.in := in
  i1.in := i0.out
  out := i1.out
}

class InOutTop extends Module {
  val definition = Definition(new TwoInOutModules)
  val instance   = Instance(definition)
  aop.Select.allInstancesOf[InOutModule](instance).foreach { i =>
    aop.Select.ios(i).foreach { io =>
      experimental.annotate(io) {
        println("instance io: " + io.toTarget)
        Nil
      }
    }
  }
  aop.Select.allDefinitionsOf[InOutModule](instance).foreach { d =>
    aop.Select.ios(d).foreach { io =>
      experimental.annotate(io) {
        println("definition io: " + io.toTarget)
        Nil
      }
    }
  }
}
```
```scala mdoc:passthrough
println("```")
val y = circt.stage.ChiselStage.emitCHIRRTL(new InOutTop)
println("```")
```
