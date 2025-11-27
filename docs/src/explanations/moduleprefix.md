# Module Prefixing

Chisel supports a feature called module prefixing.
Module prefixing allows you to create namespaces in the Verilog output of your design.
They are especially useful for when you want to name a particular subsystem of your design,
and you want to make it easy to identify which subsystem a file belongs to by its name.

## withModulePrefix

We can open a module prefix block using `withModulePrefix`:

```scala mdoc:silent
import chisel3._

class Top extends Module {
  withModulePrefix("Foo") {
    // ...
  }
}
```

All modules defined inside of this block, whether an immediate submodule or a descendent, will be given a prefix `Foo`.
(The prefix is separated by an underscore `_`).

For example, suppose we write the following:

```scala mdoc:silent:reset
import chisel3._

class Top extends Module {
  val sub = withModulePrefix("Foo") {
    Module(new Sub)
  }
}

class Sub extends Module {
  // ..
}
```

The result will be a design with two module definitions: `Top` and `Foo_Sub`.

Note that the `val sub =` part must be pulled outside of the `withModulePrefix` block,
or else the module will not be accessible to the rest of the `Top` module.

You can omit the prefix separator (`_`) by passing `false` as the second argument:

```scala mdoc:silent:reset
import chisel3._

class Top extends Module {
  val sub = withModulePrefix("Foo", false) {
    Module(new Sub)
  }
}

class Sub extends Module {
  // ..
}
```

This results in two module definitions: `Top` and `FooSub`.

## localModulePrefix

We can also set a module prefix on a module by overriding the `localModulePrefix` method.
This is useful if you want to set a prefix for all instances of a module.

```scala mdoc:silent:reset
import chisel3._

class Top extends Module {
  override def localModulePrefix = Some("Foo")
  val sub = Module(new Sub)
}

class Sub extends Module {
  // ..
}
```

This results in two module definitions: `Foo_Top` and `Foo_Sub`.

You can also override `localModulePrefixAppliesToSelf` to `false` to only apply the prefix to the children.

```scala mdoc:silent:reset
import chisel3._

class Top extends Module {
  override def localModulePrefix = Some("Foo")
  override def localModulePrefixAppliesToSelf = false
  val sub = Module(new Sub)
}

class Sub extends Module {
  // ..
}
```

This results in the two module definitions `Top` and `Foo_Sub`.

You can also override `localModulePrefixUseSeparator` to `false` to omit the separator.

```scala mdoc:silent:reset
import chisel3._

class Top extends Module {
  override def localModulePrefix = Some("Foo")
  override def localModulePrefixUseSeparator = false
  val sub = Module(new Sub)
}

class Sub extends Module {
  // ..
}
```

This results in the two module definitions `FooTop` and `FooSub`.

## Multiple Prefixes

If a generator is run in multiple prefix blocks, the result is multiple identical copies of the module definition,
each with its own distinct prefix.
For example, consider if we create two instances of `Sub` above like this:

```scala mdoc:silent:reset
import chisel3._

class Top extends Module {
  val foo_sub = withModulePrefix("Foo") {
    Module(new Sub)
  }

  val bar_sub = withModulePrefix("Bar") {
    Module(new Sub)
  }
}

class Sub extends Module {
  // ..
}
```

Then, the resulting Verilog will have three module definitions: `Top`, `Foo_Sub`, and `Bar_Sub`.
Both `Foo_Sub` and `Bar_Sub` will be identical to each other.

## Nested Prefixes

Module prefixes can also be nested.

```scala mdoc:silent:reset
import chisel3._

class Top extends Module {
  val mid = withModulePrefix("Foo") {
    Module(new Mid)
  }
}

class Mid extends Module {
  // You can mix withModulePrefix and localModulePrefix.
  override def localModulePrefix = Some("Bar")
  override def localModulePrefixAppliesToSelf = false
  val sub = Module(new Sub)
}

class Sub extends Module {
  // ..
}
```

This results in three module definitions: `Top`, `Foo_Mid`, and `Foo_Bar_Sub`.

## Instantiate

The `withModulePrefix` blocks also work with the `Instantiate` API.

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, Instantiate}

@instantiable
class Sub extends Module {
  // ...
}

class Top extends Module {
  val foo_sub = withModulePrefix("Foo") {
    Instantiate(new Sub)
  }

  val bar_sub = withModulePrefix("Bar") {
    Instantiate(new Sub)
  }

  val noprefix_sub = Instantiate(new Sub)
}
```

In this example, we end up with four modules: `Top`, `Foo_Sub`, `Bar_Sub`, and `Sub`.

When using `Definition` and `Instance`, all `Definition` calls will be affected by `withModulePrefix`.
However, `Instance` will not be effected, since it always creates an instance of the captured definition.

## External Modules

`BlackBox` and `ExtModule` are unaffected by `withModulePrefix`.
If you wish to have one that is sensitive to the module prefix,
you can explicitly name the module like this:

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.hierarchy.{instantiable, Instantiate}

class Sub extends ExtModule {
  override def desiredName = modulePrefix + "Sub"
}
```
