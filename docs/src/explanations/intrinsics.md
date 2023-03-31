---
layout: docs
title:  "Intrinsics"
section: "chisel3"
---

# Intrinsics

Chisel *Intrinsics* are used to instantiate implementation defined functionality. 
Intrinsics provide a way for specific compilers to extend the capabilities of
the language in ways which are not implementable with library code.

Modules defined as a `IntModule` will be instantiated as normal modules, but the
intrinsic field communicates to the compile what functionality to use to 
implement the module.  Implementations may not be as modules, the module nature
of intrinsics

Intrinsics will be typechecked by the implementation.  What intrinsics are 
available is documented by an implementation.

### Parameterization

Parameters can be passed as an argument to the IntModule constructor.

