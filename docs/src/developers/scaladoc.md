---
layout: docs
title:  "Scaladoc"
section: "chisel3"
---

## Scaladoc

We write inline documentation of code and public APIs using ScalaDoc.

When creating a new feature, you can view the documentation you've written by building the ScalaDoc locally with `mill`:

```sh
./mill unipublish[2.13].docJar
```

This will build a documentation jar: `out/unipublish/2.13/docJar.dest/out.jar`.
You can extract it to a directory with:
```sh
unzip out/unipublish/2.13/docJar.dest/out.jar -d apidocs
```
You can then view them by opening `apidocs/index.html` in your web browser.
