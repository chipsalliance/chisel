//===- Seq.td - Seq dialect definition ---------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the Seq dialect. It contains the one op and
// pass. Once we add more than one, we should break it out like the other
// dialects.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Seq/SeqAttributes.td"
include "circt/Dialect/Seq/SeqOpInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

// Base class for the operation in this dialect.
class SeqOp<string mnemonic, list<Trait> traits = []> :
    Op<SeqDialect, mnemonic, traits>;

def CompRegOp : SeqOp<"compreg",
    [Pure, Clocked, Resettable,
     AllTypesMatch<["input", "data"/*, "resetValue"*/]>,
     SameVariadicOperandSize,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>]> {
       // AllTypesMatch doesn't work with Optional types yet.

  let summary = "Register a value, storing it for one cycle";
  let description = "See the Seq dialect rationale for a longer description";

  let arguments = (ins AnyType:$input, I1:$clk, StrAttr:$name,
    Optional<I1>:$reset, Optional<AnyType>:$resetValue,
    OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs AnyType:$data);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    /// Create a register with an inner_sym matching the register's name.
    OpBuilder<(ins "Value":$input, "Value":$clk, "StringRef":$name), [{
      auto nameAttr = StringAttr::get($_builder.getContext(), name);
      return build($_builder, $_state, input.getType(), input, clk, nameAttr,
                   Value(), Value(), hw::InnerSymAttr::get(nameAttr));
    }]>,
    /// Create a register with a reset, with an inner_sym matching the 
    /// register's name.
    OpBuilder<(ins "Value":$input, "Value":$clk, "Value":$reset,
                   "Value":$rstValue, "StringRef":$name), [{
      auto nameAttr = StringAttr::get($_builder.getContext(), name);
      return build($_builder, $_state, input.getType(), input, clk, nameAttr,
                   reset, rstValue, hw::InnerSymAttr::get(nameAttr));
    }]>,
  ];
}

def CompRegClockEnabledOp : SeqOp<"compreg.ce",
    [Pure, Clocked, Resettable,
     AllTypesMatch<["input", "data"/*, "resetValue"*/]>,
     SameVariadicOperandSize,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>]> {
       // AllTypesMatch doesn't work with Optional types yet.

  let summary = "When enabled, register a value";
  let description = "See the Seq dialect rationale for a longer description";

  let arguments = (ins AnyType:$input, I1:$clk, I1:$clockEnable, StrAttr:$name,
    Optional<I1>:$reset, Optional<AnyType>:$resetValue,
    OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs AnyType:$data);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$clk, "Value":$ce,
                   "StringRef":$name), [{
      auto nameAttr = StringAttr::get($_builder.getContext(), name);
      return build($_builder, $_state, input.getType(), input, clk, ce,
                   nameAttr, Value(), Value(), hw::InnerSymAttr::get(nameAttr));
    }]>,
    OpBuilder<(ins "Value":$input, "Value":$clk, "Value":$ce,
                   "Value":$reset, "Value":$rstValue, "StringRef":$name),
    [{
      auto nameAttr = StringAttr::get($_builder.getContext(), name);
      return build($_builder, $_state, input.getType(), input, clk, ce,
                   nameAttr, reset, rstValue, hw::InnerSymAttr::get(nameAttr));
    }]>,
  ];
}

//===----------------------------------------------------------------------===//
// FIRRTL-flavored register
//===----------------------------------------------------------------------===//

def FirRegOp : SeqOp<"firreg",
    [Pure, Clocked, Resettable,
     AllTypesMatch<["next", "data"/*, "resetValue"*/]>,
     SameVariadicOperandSize, MemoryEffects<[MemWrite, MemRead, MemAlloc]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>]> {
      // AllTypesMatch doesn't work with Optional types yet.

  let summary = "Register with preset and sync or async reset";
  let description = [{
    `firreg` represents registers originating from FIRRTL after the lowering
    of the IR to HW.  The register is used as an intermediary in the process
    of lowering to SystemVerilog to facilitate optimisation at the HW level,
    compactly representing a register with a single operation instead of
    composing it from register definitions, always blocks and if statements.

    The `data` output of the register accesses the value it stores.  On the
    rising edge of the `clk` input, the register takes a new value provided
    by the `next` signal.  Optionally, the register can also be provided with
    a synchronous or an asynchronous `reset` signal and `resetValue`, as shown
    in the example below.

    ```
    %name = seq.firreg %next clock %clk [ sym @sym ]
        [ reset (sync|async) %reset, %value ]
        [ preset value ] : type
    ```

    Implicitly, all registers are pre-set to a randomized value.

    A register implementing a counter starting at 0 from reset can be defined
    as follows:

    ```
    %zero = hw.constant 0 : i32
    %reg = seq.firreg %next clock %clk reset sync %reset, %zero : i32
    %one = hw.constant 1 : i32
    %next = comb.add %reg, %one : i32
    ```
  }];

  let arguments = (ins AnyType:$next, I1:$clk, StrAttr:$name,
                       OptionalAttr<InnerSymAttr>:$inner_sym,
                       OptionalAttr<APIntAttr>:$preset,
                       Optional<I1>:$reset, Optional<AnyType>:$resetValue,
                       UnitAttr:$isAsync);
  let results = (outs AnyType:$data);

  let skipDefaultBuilders = 1;
  let hasCanonicalizeMethod = true;
  let hasFolder = true;
  let builders = [
    OpBuilder<(ins "Value":$next, "Value":$clk,
                   "StringAttr":$name,
                   CArg<"hw::InnerSymAttr", "{}">:$innerSym)>,
    OpBuilder<(ins "Value":$next, "Value":$clk,
                   "StringAttr":$name,
                   "Value":$reset, "Value":$resetValue,
                   CArg<"hw::InnerSymAttr", "{}">:$innerSym,
                   CArg<"bool", "false">:$isAsync)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Check whether the register has a reset value.
    bool hasReset() { return !!getReset(); }
    /// Check whether the register has a preset value.
    bool hasPresetValue() { return !!getPresetAttr(); }
  }];
}

def FIFOOp : SeqOp<"fifo", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    TypesMatchWith<"Input type should match output type",
      "input", "output", [{ $_self }]>,
    AttrSizedResultSegments
  ]> {
  let summary = "A high-level FIFO operation";
  let description = [{
    This operation represents a high-level abstraction of a FIFO. Access to the
    FIFO is structural, and thus may be composed with other core RTL dialect
    operations.
    The fifo operation is configurable with the following parameters:
    1. Depth (cycles)
    2. Almost full/empty thresholds (optional). If not provided, these will
       be asserted when the FIFO is full/empty.

    Like `seq.hlmem` there are no guarantees that all possible fifo configuration
    are able to be lowered. Available lowering passes will pattern match on the
    requested fifo configuration and attempt to provide a legal lowering.
  }];

  let arguments = (ins
    HWIntegerType:$input, I1:$rdEn, I1:$wrEn, I1:$clk, I1:$rst,
    ConfinedAttr<I64Attr, [IntMinValue<1>]>:$depth,
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>>:$almostFullThreshold,
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<0>]>>:$almostEmptyThreshold
  );

  let results = (outs
    HWIntegerType:$output, I1:$full, I1:$empty, Optional<I1>:$almostFull,
    Optional<I1>:$almostEmpty);

  let assemblyFormat = [{
    `depth` $depth
    custom<FIFOAFThreshold>($almostFullThreshold, type($almostFull))
    custom<FIFOAEThreshold>($almostEmptyThreshold, type($almostEmpty))
    `in` $input `rdEn` $rdEn `wrEn` $wrEn `clk` $clk `rst` $rst attr-dict `:` type($input)
  }];
  let hasVerifier = 1;
}

def HLMemOp : SeqOp<"hlmem", [
     Symbol,
     Clocked,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]> ]> {

  let summary = "Instantiate a high-level memory.";
  let description = "See the Seq dialect rationale for a longer description";

  let arguments = (ins I1:$clk, I1:$rst, SymbolNameAttr:$sym_name);
  let results = (outs HLMemType:$handle);

  let extraClassDeclaration = [{
    HLMemType getMemType() { return getHandle().getType().cast<HLMemType>(); }
  }];

  let builders = [
    OpBuilder<(ins "Value":$clk, "Value":$rst, "StringRef":$symName,
                   "llvm::ArrayRef<int64_t>":$shape, "Type":$elementType)>
  ];

  let assemblyFormat = "$sym_name $clk `,` $rst attr-dict `:` type($handle)";
}

class HLMemTypeValueConstraint<string hlmemvalue, string value>
  : TypesMatchWith<"type of 'value' matches element type of 'hlmemvalue'",
                     hlmemvalue, value,
                     "$_self.cast<HLMemType>().getElementType()">;

class HLMemTypeIndexingConstraint<string hlmemvalue, string addresses>
  : PredOpTrait<"address bit widths should be identical to the clog2 of the shape of the HLMemType",
      CPred<"isValidIndexValues($" # hlmemvalue # ", $" # addresses # ")">>;

def ReadPortOp : SeqOp<"read", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    HLMemTypeIndexingConstraint<"memory", "addresses">,
    AttrSizedOperandSegments     
  ]> {
  let summary = "Structural read access to a seq.hlmem, with an optional read enable signal.";
  let arguments = (ins 
    HLMemType:$memory,
    Variadic<HWIntegerType>:$addresses,
    Optional<I1>:$rdEn,
    I64Attr:$latency
  );
  let results = (outs AnyType:$readData);
  let hasCustomAssemblyFormat = 1;
  let builders = [OpBuilder<(
    ins "Value":$memory, "ValueRange":$addresses, "Value":$rdEn, "unsigned":$latency)>
  ];
}

def WritePortOp : SeqOp<"write", [
    HLMemTypeValueConstraint<"memory", "inData">,
    HLMemTypeIndexingConstraint<"memory", "addresses">
  ]> {
  let summary = "Structural write access to a seq.hlmem";
  let arguments = (ins
    HLMemType:$memory,
    Variadic<HWIntegerType>:$addresses,
    AnyType:$inData,
    I1:$wrEn,
    I64Attr:$latency
  );
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Clock Gate
//===----------------------------------------------------------------------===//

def ClockGateOp : SeqOp<"clock_gate", [Pure]> {
  let summary = "Safely gates a clock with an enable signal";
  let description = [{
    The `seq.clock_gate` enables and disables a clock safely, without glitches,
    based on a boolean enable value. If the enable operand is 1, the output
    clock produced by the clock gate is identical to the input clock. If the
    enable operand is 0, the output clock is a constant zero.

    The `enable` operand is sampled at the rising edge of the input clock; any
    changes on the enable before or after that edge are ignored and do not
    affect the output clock.

    The `test_enable` operand is optional and if present is OR'd together with
    the `enable` operand to determine whether the output clock is gated or not.

    ```
    %gatedClock = seq.clock_gate %clock, %enable : i1
    %gatedClock = seq.clock_gate %clock, %enable, %test_enable : i1, i1
    ```
  }];

  let arguments = (ins I1:$input, I1:$enable, Optional<I1>:$test_enable);
  let results = (outs I1:$output);
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
  let assemblyFormat = [{
    $input `,` $enable (`,` $test_enable^)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// FIRRTL-flavored memory
//===----------------------------------------------------------------------===//

def FirMemOp : SeqOp<"firmem", [
  MemoryEffects<[MemAlloc]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>
]> {
  let summary = "A FIRRTL-flavored memory";
  let description = [{
    The `seq.firmem` op represents memories lowered from the FIRRTL dialect. It
    is used to capture some of the peculiarities of what FIRRTL expects from
    memories, while still representing them at the HW dialect level.

    A `seq.firmem` declares the memory and captures the memory-level parameters
    such as width and depth or how read/write collisions are resolved. The read,
    write, and read-write ports are expressed as separate operations that take
    the declared memory as an operand.
  }];

  let arguments = (ins
    I32Attr:$readLatency,
    I32Attr:$writeLatency,
    RUWAttr:$ruw,
    WUWAttr:$wuw,
    OptionalAttr<StrAttr>:$name,
    OptionalAttr<InnerSymAttr>:$inner_sym,
    OptionalAttr<FirMemInitAttr>:$init,
    OptionalAttr<StrAttr>:$prefix,
    OptionalAttr<AnyAttr>:$output_file
  );
  let results = (outs FirMemType:$memory);

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name)
    $readLatency `,` $writeLatency `,` $ruw `,` $wuw
    attr-dict `:` type($memory)
  }];
}

class AddressMatchesFirMem<string memoryValue, string addressValue> :
  TypesMatchWith<"address type should match clog2 of memory depth",
    memoryValue, addressValue, [{
      IntegerType::get(
        $_self.getContext(),
        std::max(1U, llvm::Log2_64_Ceil(cast<FirMemType>($_self).getDepth())))
    }]>;

class DataMatchesFirMem<string memoryValue, string dataValue> :
  TypesMatchWith<"data type should match memory width",
    memoryValue, dataValue, [{
      IntegerType::get(
        $_self.getContext(),
        std::max(1U, cast<FirMemType>($_self).getWidth()))
    }]>;

def FirMemReadOp : SeqOp<"firmem.read_port", [
  MemoryEffects<[MemRead]>,
  AddressMatchesFirMem<"memory", "address">,
  DataMatchesFirMem<"memory", "data">
]> {
  let summary = "A memory read port";
  let description = [{
    The `seq.firmem.read_port` op represents a read port on a `seq.firmem`
    memory. It takes the memory as an operand, together with the address to
    be read, the clock on which the read is synchronized, and an optional
    enable. Omitting the enable operand has the same effect as passing a
    constant `true` to it.
  }];

  let arguments = (ins
    FirMemType:$memory,
    AnySignlessInteger:$address,
    I1:$clock,
    Optional<I1>:$enable
  );
  let results = (outs AnySignlessInteger:$data);
  let assemblyFormat = [{
    $memory `[` $address `]` `,` `clock` $clock
    (`enable` $enable^)?
    attr-dict `:` type($memory)
  }];
  let hasCanonicalizeMethod = 1;
}

def FirMemWriteOp : SeqOp<"firmem.write_port", [
  MemoryEffects<[MemWrite]>,
  AddressMatchesFirMem<"memory", "address">,
  DataMatchesFirMem<"memory", "data">,
  AttrSizedOperandSegments
]> {
  let summary = "A memory write port";
  let description = [{
    The `seq.firmem.write_port` op represents a write port on a `seq.firmem`
    memory. It takes the memory as an operand, together with the address and
    data to be written, the clock on which the write is synchronized, an
    optional enable, and and optional write mask. Omitting the enable operand
    has the same effect as passing a constant `true` to it. Omitting the write
    mask operand has the same effect as passing an all-ones value to it. A write
    mask operand can only be present if the `seq.firmem` specifies a mask width;
    otherwise it must be omitted.
  }];

  let arguments = (ins
    FirMemType:$memory,
    AnySignlessInteger:$address,
    I1:$clock,
    Optional<I1>:$enable,
    AnySignlessInteger:$data,
    Optional<AnySignlessInteger>:$mask
  );
  let assemblyFormat = [{
    $memory `[` $address `]` `=` $data `,` `clock` $clock
    (`enable` $enable^)? (`mask` $mask^)?
    attr-dict `:` type($memory) (`,` type($mask)^)?
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def FirMemReadWriteOp : SeqOp<"firmem.read_write_port", [
  MemoryEffects<[MemRead, MemWrite]>,
  AddressMatchesFirMem<"memory", "address">,
  DataMatchesFirMem<"memory", "writeData">,
  DataMatchesFirMem<"memory", "readData">,
  AttrSizedOperandSegments
]> {
  let summary = "A memory read-write port";
  let description = [{
    The `seq.firmem.read_write_port` op represents a read-write port on a
    `seq.firmem` memory. It takes the memory as an operand, together with the
    address and data to be written, a mode operand indicating whether the port
    should perform a read (`mode=0`) or a write (`mode=1`), the clock on which
    the read and write is synchronized, an optional enable, and and optional
    write mask. Omitting the enable operand has the same effect as passing a
    constant `true` to it. Omitting the write mask operand has the same effect
    as passing an all-ones value to it. A write mask operand can only be present
    if the `seq.firmem` specifies a mask width; otherwise it must be omitted.
  }];

  let arguments = (ins
    FirMemType:$memory,
    AnySignlessInteger:$address,
    I1:$clock,
    Optional<I1>:$enable,
    AnySignlessInteger:$writeData,
    I1:$mode,
    Optional<AnySignlessInteger>:$mask
  );
  let results = (outs AnySignlessInteger:$readData);
  let assemblyFormat = [{
    $memory `[` $address `]` `=` $writeData `if` $mode `,` `clock` $clock
    (`enable` $enable^)? (`mask` $mask^)?
    attr-dict `:` type($memory) (`,` type($mask)^)?
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}
