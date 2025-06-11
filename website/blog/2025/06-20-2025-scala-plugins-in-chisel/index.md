---
authors:
  - adkian-sifive
tags: [kb]
slug: scala-plugins-in-chisel
description: Scala compiler plugins in the Chisel programming language
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
behind Chisel's compiler plugins, thereby providing a much-needed
introductory treatment about them. Further, since the patterns
documented here have emerged as solutions to more general problems
faced by eDSLs, it is hoped that this documentation will serve as a
template for applying them in other DSLs and eDSLs.

No background knowledge of compiler theory or development is
assumed; however an understanding of basic Scala constructs will be
helpful in undestanding the plugin implementation examples towards the
end of the article. Introduction points for further study about
interesting points will be provided as /asides/.

# Compiler plugins

## Compilers

It seems prudent when talking about compiler plugins to start with a
brief description of /compilers/. The simplest and most general (and
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
> /themselves/ -- specifically, a compiler version x will compile the
> compiler version of x+1 (or x+0.1). See ["Wikipedia -
> Bootstrapping"](https://en.wikipedia.org/wiki/Bootstrapping_(compilers))

While in theory it's certainly possible for the compiler to go from
zero to hero in a single shot -- that is, transform straight from its
input to the output -- the complexity of modern compilers makes
compiler development more realistic by breaking up compilation into
separate chunks. These chunks are /phases/, which are in turn made of up
separate /transforms/. A phase, or a combination of phases, make up a
/pass/ -- a top to bottom transformation of the entire input program.

Each transform within each phase "reads" its input program and
rewrites it in some predetermined manner before passing it to the next
phase or transform. The earlier phases, which are responsible for
reading the syntax of the program, create what is known as an
/abstract syntax tree/, or an AST. This is basically a directed tree
representation of the entire program. The structure of the AST remains
the same for the most part; compiler passes add or remove nodes or
information from the AST to make progress towards the final output. 

## Compilers and eDSLs

An "embedded domain-specific language" (or eDSL) is a fairly new term
for an old idea -- a compiler which compiles to a higher level
language from an even higher-level, domain-specific language. The
"domains" here range from floating point operations^1^, database
management^2^, or even hardware generation^3, 4^.

General treatment of the theory of eDSL is still fairly lacking
compared to the theory of programming languages proper, but a
pertinent concept that's often mentioned is the "depth" of embedded of
an eDSL^5^. Depending on how languages DSLs are embedded in their host
language, eDSLs can either be "deep" or "shallow" embeddedings. A
shallowly embedded language is little more than a library -- import
constructs from the language and you're good to go. On the other hand,
a deeply embedded languages uses advanced metaprogramming or custom
compiler passes to append to the host language's parser or the type
system to provide a richer language interface.

One of the key factors pertinent to our current discussion which the
depth of embedding of DSLs influences is the ordering of execution of
the host language compiler and the DSL compiler. For a shallowly
embedded language, the host language compiler always executes fully before
the DSL compiler kicks in. The deeper the DSL is embedded, the earlier
its compiler kicks in -- its phases can be interspersed with those of
the host language with custom type or even syntax processing happening
before that of the host language

The Chisel programming language is a shallow embedding in the Scala
language -- that is, it is for the most part a Scala library that
allows users to import special constructs for hardware
generation. Barring a couple exceptions (that happen to be the crux of
this article), the Chisel compiler runs during "Chisel-time", a time
that is always after "Scala-time". This is a key concept for
motivating the naming problem in Chisel and other eDSLs.

## Naming problem in eDSLs

Embedded DSLs have a fundamental blind spot - variable naming. When
writing code in a eDSL, a user might write something like

```
val x = func(42)

```

thereby binding the name `x` to a variable in the current scope to the
invocation of `func`. If this compiles fine, the host language
compiler will probably transform the bound name to the value it
evaluates on the right-hand side.

Depending on the DSL, however, the user very well might expect some
bound name `x` in the final output of the DSL compiler. This, though,
will not happen if the host language erases the name or changes it to
some temporary contextual name depending on its position in the call
stack, as is common in most compiled languages. The name `x` will
hence be lost by the time the host compiler finishes, and before the
DSL compiler runs.

Therein lies the problem.

# Naming with compiler plugin
