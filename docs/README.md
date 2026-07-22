# Chisel Documentation

This directory contains documentation on the code within this repository.
Documents can either be written directly in markdown, or
use embedded [mdoc](https://scalameta.org/mdoc/)
which compiles against the `chisel` (and dependencies) codebase
as part of the PR CI checks,
forcing the documentation to remain current with the codebase.
The `src` folder contains the source from which these are generated.

Our documentation is organized into the four categories as described in
[Divio's documentation system](https://documentation.divio.com/).

The four documentation types are:
 1. Reference (source code scaladoc)
 1. Explanation (`src/explanations`)
 1. How-To Guides (`src/cookbooks`)
 1. Tutorials (currently not located here)

Our documentation strategy for this repository is as follows:
 * Any new public API requires reference documentation.
 * Any new user-facing feature requires explanation documentation.
 * Any bugfixes, corner-cases, or answers to commonly asked questions requires a how-to guide.
 * Tutorials are kept in a separate repository.

## Where to put documentation

Markdown sources live under `src`, organized by documentation type:

| Directory | Contents |
| --- | --- |
| `src/explanations` | Explanation of a feature or concept |
| `src/cookbooks` | How-to guides answering a specific question |
| `src/appendix` | Supplementary material |
| `src/developers` | Documentation for Chisel developers rather than users |
| `src/resources` | Additional resources |
| `src/images` | Images referenced by other pages |

If you create a *new* document page, you probably also want to:
  1. Add it to the "Contents" page for the corresponding directory, for example
     [cookbooks](src/cookbooks.md) or [explanations](src/explanations.md).
  1. Add it to the website sidebar in [`website/sidebars.js`](../website/sidebars.js).

## mdoc

### Prerequisites

In addition to the usual Chisel development dependencies described in
[CONTRIBUTING.md](../CONTRIBUTING.md), building the documentation requires:

 * [Scala CLI](https://scala-cli.virtuslab.org/install), used while resolving
   the firtool version associated with each Chisel release.
 * [Verilator](https://verilator.org), because some pages elaborate and
   simulate a design as part of their mdoc blocks.

Both must be on your `PATH`. Without them the build fails partway through with
`Cannot run program "scala-cli"` or `verilator not found on the PATH!`.

### Basic Use

To build the documentation, run `./mill mdoc` in the root directory.
The generated documents will appear in the `docs/generated` folder.
For more `mdoc` instructions you can visit their
[website](https://scalameta.org/mdoc/).

### Custom `verilog` modifier

mdoc supports [custom modifiers](https://scalameta.org/mdoc/docs/modifiers.html#postmodifier).
We have created a custom `verilog` modifier to enable displaying the Verilog output of Chisel.

Example use:
````
```scala mdoc:silent
class MyModule extends RawModule {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))
  out := in + 1.U
}
```
```scala mdoc:verilog
ChiselStage.emitSystemVerilog(new MyModule)
```
````
The `verilog` modifier tells mdoc to run the Scala block, requiring that each Statement returns a String.
It will then concatenate the resulting Strings and wrap them in triple backticks with the language set to `verilog`:
````
```scala
class MyModule extends RawModule {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))
  out := in + 1.U
}
```
```verilog
module MyModule(
  input  [7:0] in,
  output [7:0] out
);
  assign out = in + 8'h1;
endmodule
```
````

Note that `imports` are okay in `mdoc:verilog` blocks, but any utility Scala code should be in a separate block.

## Website

This documentation is published on the Chisel
[website](https://www.chisel-lang.org), which is built with
[Docusaurus](https://docusaurus.io/) from the [`website`](../website)
directory in this repository.
Building it requires Node.js and npm in addition to the mdoc prerequisites
above; see [`website/README.md`](../website/README.md) for the full details.

To render the site locally and view your changes:

```sh
cd website
make install  # only needed the first time
make serve
```

`make build` produces the static site without serving it.
