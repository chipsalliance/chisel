---
layout: docs
title:  "DataView Cookbook"
section: "chisel3"
---

# DataView Cookbook

* [How do I view a Data as a UInt or vice versa?](#how-do-i-view-a-data-as-a-uint-or-vice-versa)
* [How do I create a DataView for a Bundle has a type parameter?](#how-do-i-create-a-dataview-for-a-bundle-has-a-type-parameter)
* [How do I create a DataView for a Bundle with optional fields?](#how-do-i-create-a-dataview-for-a-bundle-with-optional-fields)
* [How do I connect a subset of Bundle fields?](#how-do-i-connect-a-subset-of-bundle-fields)
    * [How do I view a Bundle as a parent type (superclass)?](#how-do-i-view-a-bundle-as-a-parent-type-superclass)
    * [How do I view a Bundle as a parent type when the parent type is abstract (like a trait)?](#how-do-i-view-a-bundle-as-a-parent-type-when-the-parent-type-is-abstract-like-a-trait)
    * [How can I use `.viewAs` instead of `.viewAsSupertype(type)`?](#how-can-i-use-viewas-instead-of-viewassupertypetype)

## How do I view a Data as a UInt or vice versa?

Subword viewing (using concatenations or bit extractions in `DataViews`) is not yet supported.
We intend to implement this in the future, but for the time being, use regular casts
(`.asUInt` and `.asTypeOf`).

## How do I create a DataView for a Bundle has a type parameter?

Instead of using a `val`, use a `def` which can have type parameters:

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.dataview._

class Foo[T <: Data](val foo: T) extends Bundle
class Bar[T <: Data](val bar: T) extends Bundle

object Foo {
  implicit def view[T <: Data]: DataView[Foo[T], Bar[T]] = {
    DataView(f => new Bar(f.foo.cloneType), _.foo -> _.bar)
    // .cloneType is necessary because the f passed to this function will be bound hardware
  }
}
```

```scala mdoc:invisible
// Make sure this works during elaboration, not part of doc
class MyModule extends RawModule {
  val in = IO(Input(new Foo(UInt(8.W))))
  val out = IO(Output(new Bar(UInt(8.W))))
  out := in.viewAs[Bar[UInt]]
}
circt.stage.ChiselStage.emitSystemVerilog(new MyModule)
```
If you think about type parameterized classes as really being a family of different classes
(one for each type parameter), you can think about the `implicit def` as a generator of `DataViews`
for each type parameter.

## How do I create a DataView for a Bundle with optional fields?

Instead of using the default `DataView` apply method, use `DataView.mapping`:

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.dataview._

class Foo(val w: Option[Int]) extends Bundle {
  val foo = UInt(8.W)
  val opt = w.map(x => UInt(x.W))
}
class Bar(val w: Option[Int]) extends Bundle {
  val bar = UInt(8.W)
  val opt = w.map(x => UInt(x.W))
}

object Foo {
  implicit val view: DataView[Foo, Bar] =
    DataView.mapping(
      // First argument is always the function to make the view from the target
      f => new Bar(f.w),
      // Now instead of a varargs of tuples of individual mappings, we have a single function that
      // takes a target and a view and returns an Iterable of tuple
      (f, b) =>  List(f.foo -> b.bar) ++ f.opt.map(_ -> b.opt.get)
                                   // ^ Note that we can append options since they are Iterable!

    )
}
```

```scala mdoc:invisible
// Make sure this works during elaboration, not part of doc
class MyModule extends RawModule {
  val in = IO(Input(new Foo(Some(8))))
  val out = IO(Output(new Bar(Some(8))))
  out := in.viewAs[Bar]
}
circt.stage.ChiselStage.emitSystemVerilog(new MyModule)
```

## How do I connect a subset of Bundle fields?

Chisel 3 requires types to match exactly for connections.
DataView provides a mechanism for "viewing" one `Bundle` object as if it were the type of another,
which allows them to be connected.

### How do I view a Bundle as a parent type (superclass)?

For viewing `Bundles` as the type of the parent, it is as simple as using `viewAsSupertype` and providing a
template object of the parent type:

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.dataview._

class Foo extends Bundle {
  val foo = UInt(8.W)
}
class Bar extends Foo {
  val bar = UInt(8.W)
}
class MyModule extends Module {
  val foo = IO(Input(new Foo))
  val bar = IO(Output(new Bar))
  bar.viewAsSupertype(new Foo) := foo // bar.foo := foo.foo
  bar.bar := 123.U           // all fields need to be connected
}
```
```scala mdoc:verilog
circt.stage.ChiselStage.emitSystemVerilog(new MyModule)
```

### How do I view a Bundle as a parent type when the parent type is abstract (like a trait)?

Given the following `Bundles` that share a common `trait`:

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.dataview._

trait Super extends Bundle {
  def bitwidth: Int
  val a = UInt(bitwidth.W)
}
class Foo(val bitwidth: Int) extends Super {
  val foo = UInt(8.W)
}
class Bar(val bitwidth: Int) extends Super {
  val bar = UInt(8.W)
}
```

`Foo` and `Bar` cannot be connected directly, but they could be connected by viewing them both as if
they were instances of their common supertype, `Super`.
A straightforward approach might run into an issue like the following:

```scala mdoc:fail
class MyModule extends Module {
  val foo = IO(Input(new Foo(8)))
  val bar = IO(Output(new Bar(8)))
  bar.viewAsSupertype(new Super) := foo.viewAsSupertype(new Super)
}
```

The problem is that `viewAs` requires an object to use as a type template (so that it can be cloned),
but `traits` are abstract and cannot be instantiated.
The solution is to create an instance of an _anonymous class_ and use that object as the argument to `viewAs`.
We can do this like so:

```scala mdoc:silent
class MyModule extends Module {
  val foo = IO(Input(new Foo(8)))
  val bar = IO(Output(new Bar(8)))
  val tpe = new Super { // Adding curly braces creates an anonymous class
    def bitwidth = 8 // We must implement any abstract methods
  }
  bar.viewAsSupertype(tpe) := foo.viewAsSupertype(tpe)
}
```
By adding curly braces after the name of the trait, we're telling Scala to create a new concrete
subclass of the trait, and create an instance of it.
As indicated in the comment, abstract methods must still be implemented.
This is the same that happens when one writes `new Bundle {}`,
the curly braces create a new concrete subclass; however, because `Bundle` has no abstract methods,
the contents of the body can be empty.

### How can I use `.viewAs` instead of `.viewAsSupertype(type)`?

While `viewAsSupertype` is helpful for one-off casts, the need to provide a type template object
each time can be onerous.
Because of the subtyping relationship, you can use `PartialDataView.supertype` to create a
`DataView` from a Bundle type to a parent type by just providing the function to construct an
instance of the parent type from an instance of the child type.
The mapping of corresponding fields is automatically determined by Chisel to be the fields defined
in the supertype.

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.dataview._

class Foo(x: Int) extends Bundle {
  val foo = UInt(x.W)
}
class Bar(val x: Int) extends Foo(x) {
  val bar = UInt(x.W)
}
// Define a DataView without having to specify the mapping!
implicit val view = PartialDataView.supertype[Bar, Foo](b => new Foo(b.x))

class MyModule extends Module {
  val foo = IO(Input(new Foo(8)))
  val bar = IO(Output(new Bar(8)))
  bar.viewAs[Foo] := foo // bar.foo := foo.foo
  bar.bar := 123.U       // all fields need to be connected
}
```
```scala mdoc:verilog
circt.stage.ChiselStage.emitSystemVerilog(new MyModule)
```
