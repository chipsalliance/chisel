---
layout: docs
title:  "Annotations"
section: "chisel3"
---

# Annotations

`Annotation`s are metadata containers associated with zero or more "things" in a FIRRTL circuit.
Commonly, `Annotation`s are used to communicate information from Chisel to a specific, known FIRRTL `Transform`.
In this way `Annotation`s can be viewed as the "arguments" that a specific `Transform` consumes.

`Annotation`s are intended to be an implementation detail of Chisel and are not
meant to be manually constructed or interacted with directly by users.  Instead,
they are intended to be used through existing or new Chisel APIs.  E.g., the
`dontTouch` API provides a way for a user to indicate that a wire or port should
not be optimized.  This API is backed by a `DontTouchAnnotation`, but this is
hidden from Chisel users.

A list of all supported `Annotation`s is maintained [as part of documentation of
the FIRRTL Dialect on
circt.llvm.org](https://circt.llvm.org/docs/Dialects/FIRRTL/FIRRTLAnnotations/).
