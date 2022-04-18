---
layout: docs
title:  "Upgrading From Scala 2.11"
section: "chisel3"
redirect_from:
  - /chisel3/upgrading-from-scala-2-11.html
---

<!-- Prelude -->
```scala mdoc:invisible
import chisel3._
```
<!-- End Prelude -->

## Upgrading From Scala 2.11 to 2.12

**As of Chisel 3.5, support for Scala 2.11 has been dropped. This page is only relevant to Chisel versions 3.4 and earlier**

As the latest (and probably last) release of Scala 2.11 (2.11.12) was released on 2 November 2017, the time has come to deprecate support for Scala 2.11.
Chisel 3.4 is the last version of Chisel that will support Scala 2.11, so users should upgrade to Scala 2.12
This document is intended to help guide Chisel users through this process; both the "Why?" and the "How?".

### Scala Versioning

<!-- TODO, this should be discussed in a different document about "Building Chisel" or something-->

Scala versions have the following structure: `2.X.Y` where `X` is the _major version_ and `Y` is the _minor version_.
Note that while we keep the leading `2` constant, there is a project, [Dotty](https://dotty.epfl.ch/), that is slated to become Scala 3.

Scala maintains both source and binary compatiblity between minor versions, but not between major versions.
Binary compatibility is defined at the level of the Java Byte Code (the `.class` or `.jar` files compiled from `.scala`).
This means that Scala projects that support multiple major versions of Scala must be compiled and published for each supported version.
When publishing artifacts to Maven repositories, this manifests as an appendix on the _Artifact ID_.
Taking Chisel v3.3.2 as an example, the "Artifact ID" is ["chisel3_2.12"](https://search.maven.org/artifact/edu.berkeley.cs/chisel3_2.12)
for Scala 2.12, and ["chisel3_2.11"](https://search.maven.org/artifact/edu.berkeley.cs/chisel3_2.11) for Scala 2.11.

For more information, see the documentation on the Scala website:
* [Binary Compatibility of Scala Releases](https://docs.scala-lang.org/overviews/core/binary-compatibility-of-scala-releases.html)
* [Binary Compatibility for Library Authoers](https://docs.scala-lang.org/overviews/core/binary-compatibility-for-library-authors.html)


### How to Upgrade

For most users, this is as simple as changing the `scalaVersion` field in your `build.sbt`:
```scala
scalaVersion := "2.11.12"
```
Becomes
```scala
scalaVersion := "2.12.12"
```
Now, the next time you run SBT, it will be using the Scala 2.12 version of Chisel 3 (as well as any other dependencies you have).

### Common Issues

As mentioned in the [previous section](#scala-versioning), Scala does *not* maintain source compatibilty between major versions.
Put another way, sometimes they break things in backwards incompatible ways.
This section includes some common issues that Chisel users run into and how to fix them.

For complete information about changes, please see the [release notes for Scala 2.12.0](https://www.scala-lang.org/news/2.12.0/).

#### Value is not a member of chisel3.Bundle

The most common problem for Chisel users upgrading from Scala 2.11 to 2.12 is a change in Scala type inference.
This usually occurs in the context of `io` `Bundles` in `Modules`, given:
```scala mdoc:silent
class Foo extends Module {
  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
  })

  io.out := ~io.in
}
```
You may see an error that says somethign like "value out is not a member of chisel3.Bundle":
```
[error] /workspace/src/main/scala/gcd/Foo.scala:9:6: value out is not a member of chisel3.Bundle
[error]   io.out := ~io.in
[error]      ^
[error] /workspace/src/main/scala/gcd/Foo.scala:9:17: value in is not a member of chisel3.Bundle
[error]   io.out := ~io.in
[error]                 ^
[error] two errors found
```
This can be worked around by adding `-Xsource:2.11` to your `scalacOptions`.
This is most commonly set in your `build.sbt`.
For an example, see the [chisel-template's build.sbt](https://github.com/freechipsproject/chisel-template/blob/11f6ca470120908d167cb8dc3241953eb31d0acb/build.sbt#L10).


