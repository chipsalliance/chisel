# Chisel 3 Documentation

This directory contains documentation on the code within this repository.
Documents can either be written directly in markdown, or
use embedded [mdoc](https://scalameta.org/mdoc/)
which compiles against the `chisel3` (and dependencies) codebase
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

This documentation is hosted on the Chisel [website](https://www.chisel-lang.org).
Currently, edits to this repo which are backported to the most recent stable branch
(`3.4.x` at time of writing) will be picked up automatically by the
website [repository](https://github.com/freechipsproject/www.chisel-lang.org) and pushed live within
a day or so.
If you create a *new* document page, you probably also want to:
  1. Make sure to add it to the "Contents" page of the corresponding directory [cookbooks](src/cookbooks/cookbooks.md),
   [explanations](src/explanations/explanations.md), etc.
  1. Update the sidebar on the website [here](https://github.com/freechipsproject/www.chisel-lang.org/blob/master/docs/src/main/resources/microsite/data/menu.yml).

## mdoc

### Basic Use

To build the documentation, run `docs/mdoc` via SBT in the root directory. The generated documents
will appear in the `docs/generated` folder. To iterate on the documentation, you can run `docs/mdoc --watch`. For
more `mdoc` instructions you can visit their [website](https://scalameta.org/mdoc/).

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
ChiselStage.emitVerilog(new MyModule)
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
  assign out = in + 8'h1; // @[main.scala 9:13]
endmodule
```
````

Note that `imports` are okay in `mdoc:verilog` blocks, but any utility Scala code should be in a separate block.
