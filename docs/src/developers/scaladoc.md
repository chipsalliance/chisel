---
layout: docs
title:  "Scaladoc"
section: "chisel3"
---

## Scaladoc

We write inline documentation of code and public APIs using ScalaDoc.

When creating a new feature, you can view the documentation you've written by building the ScalaDoc locally with sbt:

```
sbt:chisel3> doc
```

This will build the documentation in `target/scala-2.12/unidoc` or `target/scala-2.13/unidoc`, and you can view it by opening `unidoc/index.html` in your browser.
