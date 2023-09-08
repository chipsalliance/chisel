---
layout: docs
title:  "Warnings"
section: "chisel3"
---

# Warnings

Warnings in Chisel are used for deprecating old APIs or semantics for later removal.
As a matter of good software practice, Chisel users are encouraged to treat warnings as errors with `--warnings-as-errors`;
however, the coarse-grained nature of this option can be problematic when bumping Chisel which may introduce many warnings.
See [Warning Configuration](#warning-configuration) below for techniques to help deal with large numbers of warnings.

## Warning Configuration

Inspired by `-Wconf` [in Scala](https://www.scala-lang.org/2021/01/12/configuring-and-suppressing-warnings.html),
Chisel supports fine-grain control of warning behavior via the CLI options `--warn-conf` and `--warn-conf-file`.

### Basic Operation

`--warn-conf` accepts a comma-separated sequence of `<filter>:<action>` pairs.
When a warning is hit in Chisel, the sequence of pairs are checked from left-to-right to see if the `filter` matches the warning.
The `action` associated with the first matching `filter` is the one used for the specific warning.
If no `filters` match, then the default behavior is to issue the warning.

`--warn-conf` can be specified any number of times.
Earlier uses of `--warn-conf` take priority over later ones in the same left-to-right decreasing priority as the `filters` are checked within a single `--warn-conf`.
As a mental model, the user can pretend that all `--warn-conf` arguments concatenated together (separated by `,`) into a single argument.

### Warning Configuration Files

`--warn-conf-file` accepts a file which contains the same format of `<filter>:<action>` pairs, separated by newlines.
Lines starting with `#` will be treated as comments and ignored.
`filters` are checked in decreasing priority from top-to-bottom of the file.

A single command-line can contain any number of `--warn-conf-file` and any number of `--warn-conf` arguments.
The filters from all `--warn-conf*` arguments will be applied in the same left-to-right decreasing priority order.

### Filters

The supported filters are:

* `any` - matches all warnings
* `id=<integer>` - matches warnings with the integer id
* `src=<glob>` - matches warnings when `<glob>` matches the source locator filename where the warning occurs

`id` and `src` filters can be combined with `&`.
Any filter can have at most one `id` and at most one `src` listed.
`any` cannot be combined with any other filters.

### Actions

The supported actions are:

* `:s` - suppress matching warnings
* `:w` - report matching warnings as warnings (default behavior)
* `:e` - error on matching warnings

### Examples

The following example issues a warning when elaborated normally

```scala mdoc:invisible:reset
// Some other test is clobbering the global Logger which breaks the warnings below
// Setting the output stream to the Console fixes the issue
logger.Logger.setConsole()
// Helper to throw away return value so it doesn't show up in mdoc
def compile(gen: => chisel3.RawModule, args: Array[String] = Array()): Unit = {
  circt.stage.ChiselStage.emitCHIRRTL(gen, args = args)
}
```

```scala mdoc
import circt.stage.ChiselStage.emitSystemVerilog
import chisel3._
class TooWideIndexModule extends RawModule {
  val in = IO(Input(Vec(4, UInt(8.W))))
  val idx = IO(Input(UInt(8.W))) // This index is wider than necessary
  val out = IO(Output(UInt(8.W)))

  out := in(idx)
}
compile(new TooWideIndexModule)
```

As shown in the warning, this warning is `W004` (which can be fixed [as described below](#w004-dynamic-index-too-wide)), we can suppress it with an `id` filter which will suppress all instances of this warning in the elaboration run.

```scala mdoc
compile(new TooWideIndexModule, args = Array("--warn-conf", "id=4:s"))
```

It is generally advisable to make warning suppressions as precise as possible, so we could combine this `id` filter with a `src` glob filter for just this file:

```scala mdoc
compile(new TooWideIndexModule, args = Array("--warn-conf", "id=4&src=**warnings.md:s"))
```

Finally, users are encouraged to treat warnings as errors to the extend possible,
so they should always end any warning configuration with `any:e` to elevate all unmatched warnings to errors:

```scala mdoc
compile(new TooWideIndexModule, args = Array("--warn-conf", "id=4&src=**warnings.md:s,any:e"))
// Or
compile(new TooWideIndexModule, args = Array("--warn-conf", "id=4&src=**warnings.md:s", "--warn-conf", "any:e"))
// Or
compile(new TooWideIndexModule, args = Array("--warn-conf", "id=4&src=**warnings.md:s", "--warnings-as-errors"))
```

## Warning Glossary

Chisel warnings have a unique identifier number to make them easier to lookup as well as so they can be configured as described above.

### [W001] Unsafe UInt cast to ChiselEnum

This warning occurs when casting a `UInt` to a `ChiselEnum` when there are values the `UInt` could take that are not legal states in the enumeration.
See the [ChiselEnum explanation](chisel-enum#casting) for more information and how to fix this warning.

**Note:** This is the only warning that is not currently scheduled for become an error.

### [W002] Dynamic bit select too wide

This warning occurs when dynamically indexing a `UInt` or an `SInt` with an index that is wider than necessary to address all bits in the indexee.
It indicates that some of the high-bits of the index are ignored by the indexing operation.
It can be fixed as described in the [Cookbook](../cookbooks/cookbook#dynamic-index-too-wide-narrow).

### [W003] Dynamic bit select too narrow

This warning occurs when dynamically indexing a `UInt` or an `SInt` with an index that is to small to address all bits in the indexee.
It indicates that some bits of the indexee cannot be reached by the indexing operation.
It can be fixed as described in the [Cookbook](../cookbooks/cookbook#dynamic-index-too-wide-narrow).

### [W004] Dynamic index too wide

This warning occurs when dynamically indexing a `Vec` with an index that is wider than necessary to address all elements of the `Vec`.
It indicates that some of the high-bits of the index are ignored by the indexing operation.
It can be fixed as described in the [Cookbook](../cookbooks/cookbook#dynamic-index-too-wide-narrow).

### [W005] Dynamic index too narrow

This warning occurs when dynamically indexing a `Vec` with an index that is to small to address all elements in the `Vec`.
It indicates that some elements of the `Vec` cannot be reached by the indexing operation.
It can be fixed as described in the [Cookbook](../cookbooks/cookbook#dynamic-index-too-wide-narrow).


### [W006] Extract from Vec of size 0

This warning occurs when indexing a `Vec` with no elements.
It can be fixed by removing the indexing operation for the size zero `Vec` (perhaps via guarding with an `if-else` or `Option.when`).
