---
authors:
  - jackkoenig
tags: [design patterns]
slug: arbiter-type-class
description: Type classes and hardware design.
---

# Type Classes and Hardware Design

When giving talks about Chisel, I always say that Chisel is intended for writing _reusable hardware generators_.
The thesis is that, by virtue of being embedded in Scala, Chisel can take advantage of powerful programming language features to make hardware design easier.
How to actually do this; however, is usually left as an exercise for the audience.

In this blog post, we'll explore how one might use the *type class* pattern to build Chisel generators.

## A Specialized Arbiter for a Specific Protocol

Let's first build a simple fixed-priority arbiter that works for a specific kind of client:

```scala
class ArbiterClient extends Bundle {
  val request = Bool()
  val grant = Flipped(Bool())
}

class PriorityArbiter(nClients: Int) extends Module {
  val clients = IO(Vec(nClients, Flipped(new ArbiterClient)))

  val requests = clients.map(_.request)
  val granted = PriorityEncoderOH(requests)

  for ((client, grant) <- clients.zip(granted)) {
    client.grant := grant
  }
}
```

<!-- truncate -->

`ArbiterClient` is written from the perspective of the client--`request` is in the default orientation, intended to be driven by the client, and `grant` is flipped, intended to be driven by the arbiter.

This works fine if you only ever need to arbitrate `ArbiterClient` bundles, but it doesn't generalize.
For example, how would you use this arbiter with a `ready-valid` interface like `Decoupled`?

```scala
val needToArbitrate = IO(Vec(4, Flipped(Decoupled(UInt(8.W)))))
val arbiter = Module(new PriorityArbiter(needToArbitrate.size))

for ((client, decoupled) <- arbiter.clients.zip(needToArbitrate)) {
  client.request := decoupled.valid
  decoupled.ready := client.grant
}
```

This works, but its a bit clunky.
Considering the implementation of `PriorityArbiter` is only 6 lines of Scala, it's not ideal that it takes another 4 lines to use it.

## Generalizing the Arbiter

What we really want is an arbiter that is generic to the type of the client.
For those familiar with _Object-Oriented Programming_, you might be tempted to try to use inheritance to define a common interface for clients:

```scala
trait Arbitrable {
  def request: Bool
  def grant: Bool
}

class ArbiterClient extends Bundle with Arbitrable {
  val request = Bool()
  val grant = Flipped(Bool())
}
```

However, for our example above, this would be a bit difficult.
`Decoupled` is defined within Chisel itself--how can a user make Chisel's `Decoupled` inherit from `Arbitrable`?

Instead, we can try something different.
We could make the Arbiter generic to the type of the client and then require additional function arguments that tell how extract the request and grant signals from the particular client type we are using.

```scala
class GenericPriorityArbiter[A <: Data](
    nClients: Int,
    clientType: A
  )(
    /** Function that indicates how to connect request from type A */
    requestFn: A => Bool,
    /** Function that indicates how to connect the grant from type A */
    grantFn: (A, Bool) => Unit) extends Module {
  val clients = IO(Vec(nClients, Flipped(clientType)))

  val requests = clients.map(requestFn(_))
  val granted = PriorityEncoderOH(requests)

  for ((client, grant) <- clients.zip(granted)) {
    grantFn(client, grant)
  }
}
```

You may notice this looks quite similar to the original `PriorityArbiter` in its implementation.
> NOTE: It uses two parameter lists in order to help the Scala type inferencer derive the types of the functions of the type of the client--we could do this with one parameter list but then it would require explicitly passing the type of the client.

Now we can use it for both our `ArbiterClient` and `Decoupled` interfaces.

```scala
val clients1 = IO(Vec(4, Flipped(new ArbiterClient)))
val arbiter1 = Module(
  new GenericPriorityArbiter(4, new ArbiterClient)(_.request, (c, g) => c.grant := g)
)
arbiter1.clients :<>= clients1

val clients2 = IO(Vec(4, Flipped(Decoupled(UInt(8.W)))))
val arbiter2 = Module(
  new GenericPriorityArbiter(4, Decoupled(UInt(8.W)))(_.valid, (d, g) => d.ready := g)
)
arbiter2.clients :<>= clients2
```

This is still a bit clunky--we have to pass two additional arguments to the arbiter.

Another thing to consider is that these additional arguments are the same for all instances of a given type.
For example, if we were to arbitrate another set of `Decoupled` interfaces, we would have to pass the same functions again:

```scala
val clients3 = IO(Vec(4, Flipped(Decoupled(UInt(8.W)))))
val arbiter3 = Module(
  new GenericPriorityArbiter(4, Flipped(Decoupled(UInt(8.W))))(_.valid, (d, g) => d.ready := g)
) // the two function arguments are the same as above.
arbiter2.clients :<>= clients3
```

## Introducing a Type Class

To clean this up even more, we can introduce a _type class_ that captures the "arbitrable" pattern:

```scala
trait Arbitrable[A] {
  def request(a: A): Bool
  def grant(a: A, value: Bool): Unit
}
```

Effectively, we have taken the two arguments to the arbiter and turned them into methods on the type class.
This looks similar to the proposed object-oriented version of `Arbitrable` above, but note how it is parameterized by the type of the client and each function defined in the trait accepts the client as an argument.

We can then provide instances of this type class for specific types. For example, for `ArbiterClient` and `Decoupled`:

```scala
class ArbiterClientArbitrable extends Arbitrable[ArbiterClient] {
  def request(a: ArbiterClient) = a.request
  def grant(a: ArbiterClient, value: Bool) = a.grant := value
}

class DecoupledArbitrable[T <: Data] extends Arbitrable[DecoupledIO[T]] {
  def request(a: DecoupledIO[T]) = a.valid
  def grant(a: DecoupledIO[T], value: Bool) = a.ready := value
}
```

Then, we can refactor the arbiter to use the type class so that we're getting one 'package' of functions for type A, not having to pass them individually:

```scala
class GenericPriorityArbiter[A <: Data](nClients: Int, clientType: A, arbitrable: Arbitrable[A]) extends Module {
  val clients = IO(Vec(nClients, Flipped(clientType)))

  val requests = clients.map(arbitrable.request(_))
  val granted = PriorityEncoderOH(requests)

  for ((client, grant) <- clients.zip(granted)) {
    arbitrable.grant(client, grant)
  }
}
```

This makes instantiating the arbiter a hair nicer:

```scala
val clients1 = IO(Vec(4, Flipped(new ArbiterClient)))
val arbiter1 = Module(new GenericPriorityArbiter(4, new ArbiterClient, new ArbiterClientArbitrable))
arbiter1.clients :<>= clients1

val clients2 = IO(Vec(4, Flipped(Decoupled(UInt(8.W)))))
val arbiter2 = Module(new GenericPriorityArbiter(4, Decoupled(UInt(8.W)), new DecoupledArbitrable[UInt]))
arbiter2.clients :<>= clients2
```

At least we aren't repeating logic anymore, instead we get to reuse the code for making the type class.

However, we can do even better.

## Implicit Type Class Instances

Scala has a powerful feature called **implicit resolution**.
This allows us to avoid figuring out what type class we need to instantiate at every call site.
Instead, we can define a default function to use when a specific type class is needed, and the compiler will automatically find it for us. We do this by making the argument to the function implicit, then making sure the implicit value of the type class is in scope.

Let us instantiate implicit functions to create our type class instances. This tells the compiler, "if you need a function to create an `Arbitrable[ArbiterClient]`, use this one."

```scala
// We could make a def, but since this function is the same every time, we just make this a `val`.
implicit val arbiterClientArbitrable: Arbitrable[ArbiterClient] =
  new Arbitrable[ArbiterClient] {
    def request(a: ArbiterClient) = a.request
    def grant(a: ArbiterClient, value: Bool) = a.grant := value
  }

// In chisel3.util, the type is DecoupledIO while we construct instances of it with Decoupled.
// Note that this is a def because DecoupledIO itself takes a type parameter,
// so we can't reuse the same one for every call-site.
implicit def decoupledArbitrable[T <: Data]: Arbitrable[DecoupledIO[T]] =
  new Arbitrable[DecoupledIO[T]] {
    def request(a: DecoupledIO[T]) = a.valid
    def grant(a: DecoupledIO[T], value: Bool) = a.ready := value
  }
```

Now we can refactor the arbiter to make its `arbitrable` argument `implicit`:

```scala
class GenericPriorityArbiter[A <: Data](nClients: Int, clientType: A)(implicit arbitrable: Arbitrable[A]) extends Module {
  val clients = IO(Vec(nClients, Flipped(clientType)))

  val requests = clients.map(arbitrable.request(_))
  val granted = PriorityEncoderOH(requests)

  for ((client, grant) <- clients.zip(granted)) {
    arbitrable.grant(client, grant)
  }
}
```

Now, we can instantiate the arbiter without passing the type class instance explicitly:

```scala
val clients1 = IO(Vec(4, Flipped(new ArbiterClient)))
val arbiter1 = Module(new GenericPriorityArbiter(4, new ArbiterClient))
arbiter1.clients :<>= clients1

val clients2 = IO(Vec(4, Flipped(Decoupled(UInt(8.W)))))
val arbiter2 = Module(new GenericPriorityArbiter(4, Decoupled(UInt(8.W))))
arbiter2.clients :<>= clients2
```

This is much cleaner and more readable. Even more importantly, it makes it the responsibility of the library
writer to determine how to make a certain type `Arbitrable`, not everyone who instantiates an arbiter.

Scala also has special syntax for the second, implicit argument list:

```scala
// Equivalent to:
// class GenericPriorityArbiter[A <: Data](nClients: Int, clientType: A)(implicit arbitrable: Arbitrable[A]) extends Module {
class GenericPriorityArbiter[A <: Data : Arbitrable](nClients: Int, clientType: A) extends Module {
  ...
}
```

This is equivalent to the previous definition, but is more concise.
Note that unlike the version with the implicit argument, this one does not bind a variable name for the implicit argument.

In the body of `GenericPriorityArbiter`, we can get a reference to the implicity value by calling `implicitly[Arbitrable[A]]`:
```scala
  val arbitrable = implicitly[Arbitrable[A]]
```

Note that Scala has rules for _implicit resolution_ for how to find the type class instance for a given type.
As a general rule, you should define implicit type class instances in the companion object of the type they are for, or in the companion object for the type class itself.

For example, since `DecoupledIO` is defined in Chisel itself, you could define the implicit value in the companion object for `Arbitrable`:
```scala
object Arbitrable {
  implicit def decoupledArbitrable[T <: Data]: Arbitrable[DecoupledIO[T]] = ...
}
```

For more information, see [further reading](#further-reading) below.

## Conclusion

This example only scratches the surface of what type classes can do in Chisel and Scala.
Whenever you find yourself passing functions around repeatedly, or are struggling with an inheritance pattern, think about whether a type class could capture that pattern.

### Further Reading

* Official Scala [documentation about type classes](https://docs.scala-lang.org/scala3/book/ca-type-classes.html)--make sure to click on the `Scala 2` tab since Chisel only currently supports Scala 2.
* Chisel DataView explanation's [section on Type Classes](../../docs/explanations/dataview#type-classes). In particular, check out the section on [implicit resolution](../../docs/explanations/dataview#implicit-resolution).
