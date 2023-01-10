---
layout: docs
title:  "Decoders"
section: "chisel3"
---

# Decoders

It is common in a complex design to recognize certain patterns from a big `UInt` coming from a data bus and dispatch
actions to next pipeline stage based on such observation. The circuit doing so can be called as 'decoders' such as
address decoders in a bus crossbar or instruction decoders in a CPU frontend. Chisel provides some utility class to
generate them in `util.exprimental.decode` package.

## Basic Decoders
The simplest API provided by `decoder` is essentially just a `TruthTable` encoding your desired input and output.
```scala mdoc:silent
import chisel3._
import chisel3.util.BitPat
import chisel3.util.experimental.decode._

class SimpleDecoder extends Module {
  val table = TruthTable(
    Map(
      BitPat("b001") -> BitPat("b?"),
      BitPat("b010") -> BitPat("b?"),
      BitPat("b100") -> BitPat("b1"),
      BitPat("b101") -> BitPat("b1"),
      BitPat("b111") -> BitPat("b1")
    ),
    BitPat("b0"))
  val input = IO(Input(UInt(3.W)))
  val output = IO(Output(UInt(1.W)))
  output := decoder(input, table)
}
```

## DecoderTable
When the decoded result involves multiple fields, each with its own semantics, the `TruthTable` can quickly be become
hard to maintain. The `DecoderTable` API is designed to generate decoder table from structured definitions.

The bridge from structured information to its encoding is `DecodePattern` trait. The `bitPat` member defines the input
`BitPat` in the decode truth table, and other members can be defined to contain structured information.

To generate output side of the decode truth table, the trait to use is `DecodeField`. Given an instance implementing the
`DecodePattern` object, the `genTable` method should return desired output.

```scala mdoc:silent
import chisel3.util.BitPat
import chisel3.util.experimental.decode._

case class Pattern(val name: String, val code: BigInt) extends DecodePattern {
  def bitPat: BitPat = BitPat("b" + code.toString(2))
}

object NameContainsAdd extends BoolDecodeField[Pattern] {
  def name = "name contains 'add'"
  def genTable(i: Pattern) = if (i.name.contains("add")) y else n
}
```

Then all `DecodePattern` cases can be generated or read from external sources. And with all `DecodeField` objects, the
decoder can be easily generated and output can be read by corresponding `DecodeField`s.
```scala mdoc:silent
import chisel3._
import chisel3.util.experimental.decode._

class SimpleDecodeTable extends Module {
  val allPossibleInputs = Seq(Pattern("addi", BigInt("0x2")) /* can be generated */)
  val decodeTable = new DecodeTable(allPossibleInputs, Seq(NameContainsAdd))
  
  val input = IO(Input(UInt(4.W)))
  val isAddType = IO(Output(Bool()))
  val decodeResult = decodeTable.decode(input)
  isAddType := decodeResult(NameContainsAdd)
}
```
