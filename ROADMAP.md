# Chisel Roadmap

This roadmap captures the current plans of the Chisel maintainers and can be
changed through the usual PR process.
Not every small feature add or planned bugfix is logged here, but the high-level
changes and planned direction that are relevant for those using and contributing
to the project.
Two especially notable directions on the roadmap below are:

* We are migrating Chisel to rely on, exclusively, the
  [MLIR-based FIRRTL Compiler](https://github.com/llvm/circt)
* Accelerating improvements and changes to Chisel and the FIRRTL specification


## 3.5.x (and Earlier)

- Continue to maintain binary compatibility on these releases
- Continue to backport bug fixes, new features if possible, documentation
  updates, etc.
- Deprecations are backported to ease in users migrating to next major release
  (3.6).

## 3.6.0 (Apr 15 2023)

This release is intended to provide a smooth path for the Chisel codebase to no
longer depend on the 
[Scala FIRRTL Compiler (SFC)](https://github.com/chipsalliance/firrtl)
and use the [MLIR Firrtl Compiler (MFC)](https://github.com/llvm/circt) as the
main backend. While the SFC may continue to work to consume the `.fir` produced
by Chisel, it will not be a requirement for adding new features to the FIRRTL
spec and the Chisel APIs. Thus, the 3.6 release will:

- Deprecate `Chisel._`
- Turn the deprecations from previous releases into Errors
- Support integration with [MFC](https://github.com/llvm/circt) by bringing
  `chisel-circt` in-repo
- Stop using [SFC](https://github.com/chipsalliance/firrtl) except to test
  deprecated features
- Deprecate all Chisel features that are not implemented in MFC. A nonexhaustive
  list includes:
  - Interval Types
  - Fixed Point Types
  - Injecting Aspects
  - EnumAnnotations
  - RunFirrtlTransformAnnotation
  - ChiselAnnotation/custom annotations
- Publish Roadmap


## 4.0

See the Q&A below.

## 5.0.0 (May 19 2023)

As we look forward to Chisel development, we plan to accelerate coordinated
changes with the
[FIRRTL specification](https://github.com/chipsalliance/firrtl-spec) to add new
functionality. The bump to version 5.0.0 will be a major change in the overall
versioning scheme of the Chisel project. We will switch to
[Semantic Versioning 2.0.0](https://semver.org/), which means we expect the
MAJOR version number to increment much more frequently. To make this change more
apparent, these changes will be made in the Chisel 5.0.0 release:

- The build artifact will be renamed to `org.chipsalliance.chisel`
- The repository will be renamed to github.com/chipsalliance/chisel (the
  `chisel3` URL will still work as an alias).
- All of Chisel’s internal tests will rely on MFC, not SFC, for lowering from
  firrtl
- Delete Chisel._
- The repo will be flattened with its dependencies/dependents as much as
  possible
  - Bring all things in from SFC into chisel repo that are needed(FIRRTL IR,
    Stage, etc)
  - chisel-test (or a subset of it that works with MFC) will be brought into the
    chisel repo
  - The chisel-lang website will be brought in-repo
- Only the chisel repo will be released as an artifact (as all supported
  dependents and dependencies will be flattened in)

## Beyond 5.0.0

Faster co-improvements to Chisel, MFC, the FIRRTL IR and Chisel.

## Q & A

* What happened to Chisel 4?

We want to make it clear that there is a change in development versioning and
philosophy (moving faster), so we are skipping 4 and moving straight from
chisel3 to Chisel 5.0.0.
