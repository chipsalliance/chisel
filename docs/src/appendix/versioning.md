---
layout: docs
title:  "Versioning"
section: "chisel3"
---

# Chisel Project Versioning

Chisel follows [Semantic Versioning 2.0.0](https://semver.org).
Project versions are of the form `MAJOR.MINOR.PATCH`.
An incremented `MAJOR` version means there may be backwards incompatible changes (typically after an appropriate deprecation schedule).
An incrementaled `MINOR` version means there are changes in functionality (new APIs) in a backwards compatible manner.
Importantly, Chisel maintains _binary compatibility_ between minor versions of the same major version.
For example, a project compiled against Chisel 5.0.0 can be used with future Chisel versions 5.1.0 or 5.100.2.
An incremented `PATCH` version means there are backwards compatible bug fixes.

## Old Chisel versions (3.0 - 3.6)

**As of Chisel 5.0.0, this and following sections no longer apply.**

Prior to the relese of Chisel 5.0.0, Chisel and related projects followed a versioning scheme similar to [PVP](https://pvp.haskell.org/).
Project versions were of the form `A.B.C` where `A.B` specifies the _Major_ version and `C` specifies the _Minor_ version.
Projects maintain _binary compatibility_ between minor versions of the same major version.

### Compatible Versions (Chisel 3.0 - 3.6)

Historically, various Chisel-related projects were distributed across multiple projects each with their own versioning.

Please use the following table to determine which versions of the related projects are compatible.
In particular, versions of projects in this table were compiled against the version of any dependencies listed in the same row.
For example, `chisel-iotesters` version 1.4 was compiled against `chisel3` version 3.3.

| chisel3 | chiseltest      | chisel-iotesters<sup>3</sup> | firrtl | treadle | diagrammer | firrtl-interpreter<sup>2</sup> |
| ------- | ----------      | ----------------             | ------ | ------- | ---------- | ----- |
| 3.6     | 0.6             | -                            | 1.6    | 1.6     | 1.6        | -     |
| 3.5     | 0.5<sup>4</sup> | 2.5<sup>5</sup>              | 1.5    | 1.5<sup>4</sup> | 1.5<sup>4</sup> | - |
| 3.4     | 0.3             | 1.5                          | 1.4    | 1.3     | 1.3        | 1.4 |
| 3.3     | 0.2             | 1.4                          | 1.3    | 1.2     | 1.2        | 1.3 |
| 3.2     | 0.1<sup>1</sup> | 1.3                          | 1.2    | 1.1     | 1.1        | 1.2 |
| 3.1     | -               | 1.2                          | 1.1    | 1.0     | 1.0        | 1.1 |
| 3.0     | -               | 1.1                          | 1.0    | -       | -          | 1.0 |

<sup>1</sup> chiseltest 0.1 was published under artifact name [chisel-testers2](https://search.maven.org/search?q=a:chisel-testers2_2.12) (0.2 was published under both artifact names)    
<sup>2</sup> Replaced by Treadle, in maintenance mode only since version 1.1, final version is 1.4    
<sup>3</sup> Replaced by chiseltest, final version is 2.5    
<sup>4</sup> chiseltest, treadle, and diagrammer skipped X.4 to have a consistent major version with Chisel    
<sup>5</sup> chisel-iotesters skipped from 1.5 to 2.5 to have a consistent major version with Chisel    
