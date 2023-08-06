//===- ESIChannels.td - All ESI ops related to app channels -- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is where any operations for the ESI dialect live.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_CHANNELS_TD
#define CIRCT_DIALECT_ESI_CHANNELS_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

include "circt/Dialect/HW/HWTypes.td"

include "circt/Dialect/ESI/ESIDialect.td"
include "circt/Dialect/ESI/ESIInterfaces.td"
include "circt/Dialect/ESI/ESITypes.td"

//===----------------------------------------------------------------------===//
// ESI channel type
//===----------------------------------------------------------------------===//

// Standard valid-ready signaling.
def ChannelSignalingValidReady : I32EnumAttrCase<"ValidReady", 0>;
// FIFO interface with a read latency of zero. Very similiar to valid-ready, but
// cannot assert ready if data is not available.
def ChannelSignalingFIFO0      : I32EnumAttrCase<"FIFO0",      1>;

def ChannelSignaling : I32EnumAttr<
    "ChannelSignaling",
    "ESI channel wire signaling standard",
    [ChannelSignalingValidReady, ChannelSignalingFIFO0]>{
  let cppNamespace = "::circt::esi";
}

def ChannelTypeImpl : ESI_Type<"Channel"> {
  let summary = "An ESI-compatible channel port";
  let description = [{
    An ESI port kind which models a latency-insensitive, unidirectional,
    point-to-point data stream. Channels are typed (like all of ESI). Said
    type can be any MLIR type, but must be lowered to something a backend
    knows how to output (i.e. something emitVerilog knows about).

    Example:

    ```mlir
    hw.module.extern @Sender() -> (%x: !esi.channel<i1>)
    hw.module @Reciever(%a: !esi.channel<hw.array<5xi16>>) { }
    ```
  }];

  let mnemonic = "channel";
  let parameters = (ins
      "Type":$inner,
      DefaultValuedParameter<
        "::circt::esi::ChannelSignaling",
        "::circt::esi::ChannelSignaling::ValidReady">:$signaling);

  let assemblyFormat = "`<` $inner (`,` $signaling^)? `>`";
 
  let builders = [
    TypeBuilder<(ins "Type":$type), [{
        return Base::get(type.getContext(), type,
                         ::circt::esi::ChannelSignaling::ValidReady);
      }]>,
  ];
}

//===----------------------------------------------------------------------===//
// Wrap / unwrap channels to their signaling protocols.
//===----------------------------------------------------------------------===//

def WrapValidReadyOp : ESI_Op<"wrap.vr", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Wrap a value into an ESI port";
  let description = [{
    Wrapping a value into an ESI port type allows modules to send values down
    an ESI port. Wrap data with valid bit, result is the ESI channel and the
    ready signal from the other end of the channel.
  }];

  let arguments = (ins AnyType:$rawInput, I1:$valid);
  let results = (outs ChannelType:$chanOutput, I1:$ready);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$data, "mlir::Value":$valid)>
  ];
}

def UnwrapValidReadyOp : ESI_Op<"unwrap.vr", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Unwrap a value from an ESI port";
  let description = [{
    Unwrapping a value allows operations on the contained value. Unwrap the
    channel along with a ready signal that you generate. Result is the data
    along with a valid signal.
  }];

  let arguments = (ins ChannelType:$chanInput, I1:$ready);
  let results = (outs AnyType:$rawOutput, I1:$valid);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$inChan, "mlir::Value":$ready)>
  ];
}

def WrapFIFOOp : ESI_Op<"wrap.fifo", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Wrap a value into an ESI port with FIFO signaling";

  let arguments = (ins AnyType:$data, I1:$empty);
  let results = (outs ChannelType:$chanOutput, I1:$rden);
  let hasCanonicalizeMethod = true;
  let hasFolder = true;
  let hasVerifier = 1;

  let assemblyFormat = [{
    $data `,` $empty attr-dict `:`
      custom<WrapFIFOType>(type($data), type($chanOutput))
  }];
}

def UnwrapFIFOOp : ESI_Op<"unwrap.fifo", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Unwrap a value from an ESI port into a FIFO interface";

  let arguments = (ins ChannelType:$chanInput, I1:$rden);
  let results = (outs AnyType:$data, I1:$empty);
  let hasCanonicalizeMethod = true;
  let hasVerifier = 1;

  let assemblyFormat = [{
    $chanInput `,` $rden attr-dict `:` qualified(type($chanInput))
  }];

  let extraClassDeclaration = [{
    static LogicalResult mergeAndErase(UnwrapFIFOOp, WrapFIFOOp, PatternRewriter&);
  }];
}

def ModportType:
  Type<CPred<"$_self.isa<::circt::sv::ModportType>()">, "sv.interface">;

def WrapSVInterfaceOp: ESI_Op<"wrap.iface", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Wrap an SV interface into an ESI port";
  let description = [{
    Wrap a SystemVerilog interface into an ESI channel. Interface MUST look
    like an interface produced by ESI meaning it MUST contain valid, ready,
    and data signals. Any other signals will be discarded.
  }];

  let arguments = (ins ModportType:$interfaceSink);
  let results = (outs ChannelType:$output);

  let assemblyFormat = [{
    $interfaceSink attr-dict `:` qualified(type($interfaceSink)) `->` qualified(type($output))
  }];

  let hasVerifier = 1;
}

def UnwrapSVInterfaceOp : ESI_Op<"unwrap.iface", [
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Unwrap an SV interface from an ESI port";
  let description = [{
    Unwrap an ESI channel into a SystemVerilog interface containing valid,
    ready, and data signals.
  }];

  let arguments = (ins ChannelType:$chanInput, ModportType:$interfaceSource);
  let results = (outs);

  let assemblyFormat = [{
    $chanInput `into` $interfaceSource attr-dict `:` `(` qualified(type($chanInput)) `,` qualified(type($interfaceSource)) `)`
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Data windows
//===----------------------------------------------------------------------===//

def ESIWindowType : ESI_Type<"Window"> {
  let summary = "a data window";
  let description = [{
    A 'data window' allows designers to break up large messages into multiple
    frames (aka phits) spread across multiple cycles. Windows are specified in
    terms of a mapping of struct fields to frames. The width of a window is the
    maximum frame size + the union tag (log2(#frames)).

    A data window does NOT imply an ESI channel.

    Current restrictions:
    - A field may only appear once.
    - Fields may only be re-ordered (wrt the original message) within a frame.
    - Array fields whose array length is not evenly divisible by 'numItems' will
    have an implicit frame inserted directly after containing the leftover array
    items.
    - Array fields with an array length MUST be in their own frame.
  }];

  let mnemonic = "window";
  let genVerifyDecl = 1;

  let parameters = (ins "StringAttr":$name, "mlir::Type":$into,
                        ArrayRefParameter<"WindowFrameType">:$frames);
  let assemblyFormat = [{ `<` $name `,` $into `,` `[` $frames `]` `>` }];

  let extraClassDeclaration = [{
    /// Get the union which realizes this window during lowering.
    circt::hw::UnionType getLoweredType() const;
  }];
}

def WindowFrameType : ESI_Type<"WindowFrame"> {
  let summary = "Declare a data window frame";
  let description = [{
    A named list of fields which should appear in a given frame.
  }];
  let mnemonic = "window.frame";

  let parameters = (ins "StringAttr":$name,
                        ArrayRefParameter<"WindowFieldType">:$members);

  let assemblyFormat = [{
    `<` $name `,` `[` $members `]` `>`
  }];
}

def WindowFieldType : ESI_Type<"WindowField"> {
  let summary = "a field-in-frame specifier";
  let description = [{
    Specify that a field should appear within the enclosing frame.
  }];
  let mnemonic = "window.field";

  let parameters = (ins
      "StringAttr":$fieldName,
      OptionalParameter<"uint64_t", "# of items in arrays or lists">:$numItems);

  let assemblyFormat = [{
    `<` $fieldName (`,` $numItems^)? `>`
  }];
}

def WrapWindow : ESI_Physical_Op<"window.wrap", [Pure]> {
  let summary = "wrap a union into a data window";

  let arguments = (ins UnionType:$frame);
  let results = (outs ESIWindowType:$window);
  let hasVerifier = 1;
  let hasFolder = 1;

  let assemblyFormat = [{
     $frame attr-dict `:` custom<InferWindowRet>(type($frame), type($window))
  }];
}

def UnwrapWindow : ESI_Physical_Op<"window.unwrap", [
      Pure,
      DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "unwrap a data window into a union";

  let arguments = (ins ESIWindowType:$window);
  let results = (outs UnionType:$frame);
  let hasFolder = 1;

  let assemblyFormat = [{
     $window attr-dict `:` qualified(type($window))
  }];
}

//===----------------------------------------------------------------------===//
// Channel buffering and pipelining.
//===----------------------------------------------------------------------===//

def ChannelBufferOp : ESI_Abstract_Op<"buffer", [
    Pure,
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "Control options for an ESI channel.";
  let description = [{
    A channel buffer (`buffer`) is essentially a set of options on a channel.
    It always adds at least one cycle of latency (pipeline stage) to the
    channel, but this is configurable.

    This operation is inserted on an ESI dataflow edge. It must exist
    previous to SystemVerilog emission but can be added in a lowering pass.

    A `stages` attribute may be provided to specify a specific number of cycles
    (pipeline stages) to use on this channel. Must be greater than 0.

    A `name` attribute may be provided to assigned a name to a buffered
    connection.

    Example:

    ```mlir
    %esiChan = hw.instance "sender" @Sender () : () -> (!esi.channel<i1>)
    // Allow automatic selection of options.
    %bufferedChan = esi.buffer %esiChan : i1
    hw.instance "recv" @Reciever (%bufferedChan) : (!esi.channel<i1>) -> ()

    // Alternatively, specify the number of stages.
    %fourStageBufferedChan = esi.buffer %esiChan { stages = 4 } : i1
    ```
  }];

  let arguments = (ins I1:$clk, I1:$rst, ChannelType:$input,
    OptionalAttr<ConfinedAttr<I64Attr, [IntMinValue<1>]>>:$stages,
    OptionalAttr<StrAttr>:$name);
  let results = (outs ChannelType:$output);
  let hasCustomAssemblyFormat = 1;
}

def PipelineStageOp : ESI_Physical_Op<"stage", [
    Pure,
    DeclareOpInterfaceMethods<ChannelOpInterface>
  ]> {
  let summary = "An elastic buffer stage.";
  let description = [{
    An individual elastic pipeline register. Generally lowered to from a
    ChannelBuffer ('buffer'), though can be inserted anywhere to add an
    additional pipeline stage. Adding individually could be useful for
    late-pass latency balancing.
  }];

  let arguments = (ins I1:$clk, I1:$rst, ChannelType:$input);
  let results = (outs ChannelType:$output);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Misc operations
//===----------------------------------------------------------------------===//

def CosimEndpointOp : ESI_Physical_Op<"cosim", []> {
  let summary = "Co-simulation endpoint";
  let description = [{
    A co-simulation endpoint is a connection from the simulation to some
    outside process, usually a software application responsible for driving
    the simulation (driver).

    ESI uses a serialization protocol called Cap'n Proto (capnp for short).
    The advantage of capnp is the decoding overhead: for value types (ints,
    structs, etc.) there is none! This stands in contrast to Protocol Buffers
    and Bond as their messages contain metadata for each field which must be
    interpreted.

    The advantage of using a well-supported serialization protocol is
    language support -- driver applications can be written in any language
    supported by the specific protocol.
  }];

  let arguments = (ins I1:$clk, I1:$rst, ChannelType:$send, StrAttr:$name);
  let results = (outs ChannelType:$recv);

  let assemblyFormat = [{
    $clk `,` $rst `,` $send `,` $name attr-dict
      `:` qualified(type($send)) `->` qualified(type($recv))
  }];
}

def RtlBitArrayType : Type<CPred<"$_self.isa<::circt::hw::ArrayType>()"
 " && $_self.cast<::circt::hw::ArrayType>().getElementType() =="
 "   ::mlir::IntegerType::get($_self.getContext(), 1)">, "an HW bit array">;

def CapnpDecodeOp : ESI_Physical_Op<"decode.capnp", [Pure]> {
  let summary = "Translate bits in Cap'nProto messages to HW typed data";

  let arguments = (ins I1:$clk, I1:$valid, RtlBitArrayType:$capnpBits);
  let results = (outs AnyType:$decodedData);

  let assemblyFormat = [{
    $clk $valid $capnpBits attr-dict `:` qualified(type($capnpBits)) `->`
                                         qualified(type($decodedData))
  }];
}

def CapnpEncodeOp : ESI_Physical_Op<"encode.capnp", [Pure]> {
  let summary = "Translate HW typed data to Cap'nProto";

  let arguments = (ins I1:$clk, I1:$valid, AnyType:$dataToEncode);
  let results = (outs RtlBitArrayType:$capnpBits);

  let assemblyFormat = [{
    $clk $valid $dataToEncode attr-dict `:` qualified(type($dataToEncode))
                                          `->` qualified(type($capnpBits))
  }];
}

def NullSourceOp : ESI_Physical_Op<"null", [Pure]> {
  let summary = "An op which never produces messages.";

  let arguments = (ins);
  let results = (outs ChannelType:$out);

  let assemblyFormat = [{ attr-dict `:` qualified(type($out)) }];
}

#endif // CIRCT_DIALECT_ESI_CHANNELS_TD
