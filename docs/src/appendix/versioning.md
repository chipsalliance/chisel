---
layout: docs
title:  "Versioning"
section: "chisel3"
---

# Chisel Project Versioning

Chisel and related projects follow a versioning scheme similar to [PVP](https://pvp.haskell.org/).
Project versions are of the form `A.B.C` where `A.B` specifies the _Major_ version and `C` specifies the _Minor_ version.
Projects maintain _binary compatibility_ between minor versions of the same major version.
For example, a project compiled against Chisel3 version 3.4.0 can be used with Chisel3 version 3.4.2 or 3.4.15 without recompilation.

# Compatible Versions

Historically, due to a mistake in versioning with chisel-iotesters as well as some projects originating later than others,
the compatible versions of Chisel-related projects has not been obvious.
We are taking steps to improve the situation by bringing the major versions more in line (the `B` part in `A.B.C`),
but the inconsistencies remain in previously published versions.

Please use the following table to determine which versions of the related projects are compatible.
In particular, versions of projects in this table were compiled against the version of any dependencies listed in the same row.
For example, `chisel-iotesters` version 1.4 was compiled against `chisel3` version 3.3.

| chisel3 | chiseltest | chisel-iotesters | firrtl | treadle | diagrammer | firrtl-interpreter<sup>2</sup> |
| ------- | ---------- | ---------------- | ------ | ------- | ---------- | ----- |
| 3.4     | 0.3        | 1.5              | 1.4    | 1.3     | 1.3        | 1.4 |
| 3.3     | 0.2        | 1.4              | 1.3    | 1.2     | 1.2        | 1.3 |
| 3.2     | 0.1<sup>1</sup>  | 1.3        | 1.2    | 1.1     | 1.1        | 1.2 |
| 3.1     | -          | 1.2              | 1.1    | 1.0     | 1.0        | 1.1 |
| 3.0     | -          | 1.1              | 1.0    | -<sup>3</sup> | -    | 1.0 |

<sup>1</sup> chiseltest 0.1 was published under artifact name [chisel-testers2](https://search.maven.org/search?q=a:chisel-testers2_2.12) (0.2 was published under both artifact names)    
<sup>2</sup> Replaced by Treadle, in maintenance mode only since version 1.1    
<sup>3</sup> Treadle was preceded by the firrtl-interpreter    
