---
layout: docs
title:  "Treadle"
section: "treadle"
position: 5
---

## What is the Treadle

Treadle is a hardware circuit simulator that takes its circuit description directly from FIRRTL.
It is based on earlier work done in the
[FirrtlInterpreter](https://github.com/freechipsproject/firrtl-interpreter).
Treadle is most commonly used as a backend for ChiselTest and ChiselTesters unit tests framework.
It supports a Peek, Poke, Expect, Step interface.
Treadle can be quite a bit slower for very large circuits but it spins up much faster than
other backends so for unit tests of small to medium circuits it generally runs considerably faster.

Treadle also provides the [TreadleRepl](TreadleRepl.html), an interactive shell that provides
execution and debugging commands.
