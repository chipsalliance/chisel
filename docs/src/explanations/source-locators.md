---
layout: docs
title:  "Source Locators"
section: "chisel3"
---

# Source Locators

When elaborating a Chisel design and emitting a FIRRTL file or Verilog file, Chisel will automatically
add source locators which refer back to the Scala file containing the corresponding Chisel code.

In a FIRRTL file, it looks like this:

```
wire w : UInt<3> @[src/main/scala/MyProject/MyFile.scala 1210:21]
```

In a Verilog file, it looks like this:

```verilog
wire [2:0] w; // @[src/main/scala/MyProject/MyFile.scala 1210:21]
```

By default, the file's relative path to where the JVM is invoked is included.
To change where the relative path is computed, set the Java system property `-Dchisel.project.root=/absolute/path/to/root`.
This option can be directly passed to sbt (`sbt -Dchisel.project.root=/absolute/path/to/root`).
Setting the value in the `build.sbt` file won't work because it needs to be passed to the JVM that invokes sbt (not the other way around).
We expect this only relevant for publishing versions which may want more customization.
