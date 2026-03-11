---
authors:
  - adkian-sifive
tags: [kb]
slug: scala-plugins-in-chisel
description: Scala compiler plugins for Naming in the Chisel programming language
---

# Introduction

Compiler plugins are a feature of the Scala programming language which
the Chisel eDSL uses extensively. Similar to their counterparts in
Kotlin and Haskell (as GHC plugins), they allow extending the behavior
of the compiler with custom AST inspection and transformation. While
quite a powerful tool, writing and understanding compiler plugins is
often an esoteric art -- plugin code has a tendency to be exceptionally
difficult to decipher by anyone other than its author. 

My goal with this article is to motivate and demystify the subtle art
behind Chisel's naming compiler plugin, thereby providing a much-needed
introductory treatment about it. Further, since the patterns
documented here have emerged as solutions to more general problems
faced by eDSLs, it is hoped that this documentation will serve as a
template for applying them in other DSLs and eDSLs.

No background knowledge of compiler theory or development is assumed;
however an understanding of basic Chisel and Scala constructs will be
helpful in motivating the examples in the article. Introduction points
for further study about interesting topics will be provided as
*asides*.

# Compilers

It seems prudent when talking about compiler plugins to start with a
brief description of *compilers*. The simplest and most general (and
indeed useless) description of a compiler -- including the Scala
compiler -- is that it's a piece of executable software that
transforms one representation of a piece of text to another. More
often than not, the initial representation -- the "input" -- is some
kind of human-readable program; the final representation -- the
"output" -- is a machine-executable binary. 

> **_Aside: Who compiles the compiler?_**  If the compiler is
> executable software, mustn't it also be compiled? The answer is yes
> -- the code of a compiler is also compiled by some other
> compiler. Most modern compilers are in fact able to compile
> *themselves* -- specifically, a compiler version x will compile the
> compiler version of x+1 (or x+0.1). See ["Wikipedia -
> Bootstrapping"](https://en.wikipedia.org/wiki/Bootstrapping_(compilers))

While in theory it's certainly possible for the compiler to go from
zero to hero in a single shot -- that is, transform straight from its
input to the output -- the complexity of modern compilers makes
compiler development more realistic by breaking up compilation into
separate chunks. These chunks are *phases*, which are in turn made of up
separate *transforms*. A phase, or a combination of phases, make up a
*pass* -- a top to bottom transformation of the entire input program.

Each transform within each phase "reads" its input program and
rewrites it in some predetermined manner before passing it to the next
phase or transform. The earlier phases, which are responsible for
reading the syntax of the program, create what is known as an
/abstract syntax tree/, or an AST. This is basically a directed tree
representation of the entire program. The structure of the AST remains
the same for the most part; compiler passes add or remove nodes or
information from the AST to make progress towards the final output. 

# Compilers and eDSLs

An "embedded domain-specific language" (or eDSL) is a fairly new term
for an old idea -- a compiler which compiles to a higher level
language from an even higher-level, domain-specific language. The
"domains" here range from numerical operations<sup>1</sup>, database
management<sup>2</sup>, or even... hardware generation<sup>3, 4</sup>.

General treatment of the theory of eDSLs is still fairly lacking
compared to the theory of programming languages proper, but a
pertinent concept that's often mentioned is the "depth" of embedded of
an eDSL^5^. Depending on how DSLs are embedded in their host
languages, eDSLs can either be "deep" or "shallow" embeddings. A
shallowly embedded language is little more than a library -- import
constructs from the language and you're good to go. On the other hand,
a deeply embedded language uses advanced metaprogramming or custom
compiler passes to append to the host language's parser or the type
system to provide a richer language interface.

One of the key factors relevant to our current discussion which the
depth of embedding of DSLs influences is the ordering of execution of
the host language compiler and the DSL compiler. For a shallowly
embedded language, the host language compiler always executes fully before
the DSL compiler kicks in. The deeper the DSL is embedded, the earlier
its compiler kicks in -- its phases can be interspersed with those of
the host language with custom type or even syntax processing happening
before that of the host language.

The Chisel programming language is a shallow embedding in the Scala
language -- that is, it is for the most part a Scala library that
allows users to import special constructs for hardware
generation. Barring a couple exceptions (that happen to be the crux of
this article), the Chisel compiler runs during "Chisel-time", a time
that is always after "Scala-time". This is a key concept for
motivating the naming problem in Chisel and other eDSLs.

# Compiler plugins

Various languages provide varying degrees of support of *compiler
plugins* - a feature for DSL developers that allows adding custom passes which
will be interspersed within the host language's compiler pipeline.

In Scala, compiler plugin phases run between specified phases of the
host compiler. These phases can have custom transforms that receive
ASTs from the previous phase, inspect and transform them, and send
them along to the next phase. Compiler plugins are written within the
host Scala compiler's namespace and context. While this provides them
with the full power of the compiler itself, it requires quite a bit of
Scala compiler knowledge to implement.

> **_Aside: Research plugins_** Scala 3 introduces experimental
> "Research Plugins", which allow plugin developers to completely
> rewrite the ordering of all the phases of the Scala compiler
> pipeline. See ["Scala Reference - Compiler
> Plugins"](https://docs.scala-lang.org/scala3/reference/changed-features/compiler-plugins.html)

# Naming problem in eDSLs

Embedded DSLs tend to have a fundamental blind spot - variable
naming. When writing code in an eDSL, a user might write something
like

```
val x = func(42)
```

thereby binding the name `x` to a variable in the current scope to the
invocation of `func`. If this compiles fine, the host language
compiler will probably transform the bound name to the value it
evaluates on the right-hand side.

Depending on the DSL the user can certainly expect some bound name `x`
in the final output of the compiler. This, however, will not happen
if the host language erases the name or changes it to some temporary
contextual name depending on its position in the call stack, as is
common in most compiled languages. The name `x` will hence be lost by
the time the host compiler finishes, and before the DSL compiler runs.

This means that without any intervention on the part of the DSL
developer, variable names from the input user code cannot always be
expected to be syntactically or semantically preserved in the compiled
code.

In short, variables used in a Chisel program are all native Scala
variables whose names are only available in Scala-time and not in
Chisel-time. 

Therein lies the problem.

# Solving the naming problem

## Chisel naming with compiler plugin

Predictable user-code naming is especially important in
Chisel as it compiles down to Verilog, where engineers rely on
predictable signal naming schemes from Chisel all the way down
through FIRRTL<sup>6</sup> and to Verilog for signal tracing, debugging and
hardware verification. 

> **_Aside: Naming in Chisel_** Naming of user-defined variables has
> evolved significantly in Chisel over each major version and has
> recently stabilized, with now multiple naming schemes available based
> on user requirements -- from simple name "suggestions" to the
> compiler to more complex custom name overrides. See ["Chisel
> Explanations -
> Naming"](https://www.chisel-lang.org/docs/explanations/naming)

To capture as raw of a user-code name as possible, Chisel runs a
custom name-grabber pass right after Scala finishes constructing a
typed AST. Just naively capturing variable names won't do, however,
since Chisel only cares about names of *Chisel types* which will in
turn become hardware names in the final Verilog. 

To make sure the plugin only grabs the names of Chisel types, the
transform methods in the custom naming phase inspect the right hand
side of every `val` definition -- sometimes even recursively, if
needed, for boxing types such as `Option`. 

> **_Aside: Chisel... types?_** I say "Chisel types" here which is a
> bit of a misnomer. Chisel actually doesn't have its own type system
> -- it relies on a system of objects to create instances of subtypes
> of the Data class. This system has its pros and cons and shall
> perhaps be a topic of a future article. For now, see ["Chisel
> Explanations -
> Datatypes"](https://www.chisel-lang.org/docs/explanations/data-types)

In the internal representation of the syntax tree, each AST node
refers to some statement in the language; the ones we are interested
in for variable naming are of the type `val x = func(42)`. When Scala's
parser and the typer phases have run over this statement, the internal
representation of this statement in the compiler looks something like:

```
ValDef(
  "x",
  TypeTree[TypeRef(ThisType(...))],
  Apply(
    Ident(func),
    List(Literal(Constant(42)))
  )
)
```

`ValDef`, `TypeTree`, `Apply`, `Ident` and so on are internal types of
the Scala compiler source code which can be composed to create an AST
representation.

Once a Chisel type has been detected by the naming transformation, it
inspects the syntax tree of each AST node to extract the variable name
of the current `val`. This can done pretty easily with matching over
the `ValDef` as seen above to extract the first field which is the
variable name. The naming phase now knows that there's a `val`
definition bound to "x" with a type we're interested in.

Next, we need to propagate this name from Scala-time to Chisel-time,
so that Chisel can sanitize it if necessary and lower it correctly to
FIRRTL. This is done using the `withName` method that's part of Chisel
`core`. The naming plugin rewrites the AST node of the `val`
definition to insert a call to the `withName` method on the RHS of the
val definition. The above statement will hence become:

```
val x = chisel3.withName("x")(func(42))
```

The `withName` insertion into the AST node effectively *stages* the
addition of a string name until Chisel-time, when Chisel internals
process naming given the Chisel-time context of the variable
definition statement. The Chisel-time name processing is its own
beast and deserves its own article.


## Summary: Chisel naming plugin

Here's a short top-to-bottom summary of the naming plugin:

- User writes Chisel code and compiles it with the latest Chisel
  compiler
- The Chisel compiler registers a naming phase with the Scala
  compiler, and the build tool runs the Scala compiler
- During Scala compile time, after the Scala compiler's parser and
  typer phases are finished running, the naming plugin receives the
  AST from the typer phase and runs transformations on each `ValDef`
  definition it encounters.
- For each `ValDef` AST node, it inspects the right-hand side of the
  statement to check if somewhere in the leaf nodes of the thing being
  defined is a Chisel type.
- If a Chisel type is found, it extracts the variable name from the
  left-hand side
- The plugin rewrites the RHS of the statement with a call to
  `chisel3.withName`, thereby staging the variable name of the
  statement into a method that executes at Chisel-time
- During Chisel-time, Chisel executes `withName` and names the
  variables based on the Chisel-time context at the statement

## The reality of naming: Diversity of solutions

### Chisel pre-3.4

Chisel's use of a Scala compiler plugin was a fairly recent
innovation. Previously, Chisel relied on Scala 2's "Macro Paradise"
plugin -- which itself is a compiler plugin -- to apply a
`suggestName` for each declared variable. This was considered fragile
for several reasons, including the fact that the macro invocation
would be handled in Chisel-time, not Scala-time. This meant that names
were harvested after Scala had completed all phases and names could
not necessarily be deterministically computed from inputs. 

### Other DSLs

A survey of existing implementations of eDSLs in different functional
languages seems to suggest that the naming problem is somewhat
ubiquitous. In most languages, some form of compile-time reflection is
needed to capture user code naming.

Unsurprisingly, the "purest" solution to the naming problem that
falls naturally from the host language's metaprogramming features is
in Lisp. This is because of Lisp's so-called *homoiconicity* -- its
ability to treat representation of its own code as data itself. This
means that any Lisp macro defined in the DSL implementation can *see*
the symbol that user code references as data.<sup>7</sup>

The Clash programming language, a fellow hardware eDSL implemented in
Haskell, has a slight advantage due to its deeper embedding. Clash has
access to constructs within Haskell core including the OccName<sup>8</sup> data
structure that contains plaintext name and namespace information for
every declared symbol in user code. Clash utilizes the plaintext names
from OccName and applies similar disambiguation and sanitization as
Chisel core. 

Most modern Haskell eDSLs whose compilers run separately from the GHC
tend to use Template Haskell, an experimental metaprogramming
facility.<sup>9</sup>

Interestingly enough, an older staged DSL implemented in Haskell
called Paradise came up with a solution identical to the one currently
implemented in Chisel where the GHC preprocessor inserts calls to an
annotation function coincidentally also called `withName`.<sup>10</sup>

# Final Remarks

Chisel naming has come a long way, and after undergoing heavy
utilization and customization in mission-critical applications at
[SiFive](https://sifive.com), can safely be deemed to be stable. With the
ongoing work of adding support for Scala 3 in Chisel, we're hoping to
develop cleaner and more readable Scala compiler plugins.

# References and further reading
<sup>1</sup>[Typelevel - Spire ](https://typelevel.org/spire/)

<sup>2</sup>[Apache Spark](https://spark.apache.org/)

<sup>3</sup>[Clash Programming Language](https://clash-lang.org/)

<sup>4</sup>[Spatial Programming Language](https://spatial-lang.org/)

<sup>5</sup>[Folding Domain-Specific Languages: Deep and Shallow Embeddings](https://www.cs.ox.ac.uk/jeremy.gibbons/publications/embedding.pdf)

See also [YouTube: Tiark Rompf - DSL Embedding in Scala](https://www.youtube.com/watch?v=16A1yemmx-w)

<sup>6</sup>[The FIRRTL Spec](https://github.com/chipsalliance/firrtl-spec)

<sup>7</sup>[Common Lisp CookBook](https://cl-cookbook.sourceforge.net/clos-tutorial/index.html)

<sup>8</sup>[OccName in 
Haskell](https://downloads.haskell.org/~ghc/6.10.2/docs/html/libraries/ghc/OccName.html)

<sup>9</sup>[Naming with Template Haskell](https://markkarpov.com/tutorial/th?#names)

<sup>10</sup>[Paradise: A two-stage DSL embedded in Haskell](https://urchin.earth.li/~ganesh/icfp08.pdf)
