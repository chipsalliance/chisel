//===- DCOps.td - DC dialect operations --------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DC_OPS_TD
#define CIRCT_DIALECT_DC_OPS_TD

include "mlir/Interfaces/InferTypeOpInterface.td"
include "circt/Dialect/DC/DCDialect.td"
include "circt/Dialect/DC/DCTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/FunctionInterfaces.td"
include "circt/Dialect/ESI/ESITypes.td"

class DCOp<string mnemonic, list<Trait> traits = []> :
  Op<DCDialect, mnemonic, !listconcat(traits, [
    Pure
  ])>;

class SameTypeConstraint<string lhs, string rhs>
 : TypesMatchWith<"lhs and rhs types should be equivalent",
                   lhs, rhs, [{ $_self }]>;

def BufferOp : DCOp<"buffer",
    [SameTypeConstraint<"input", "output">]> {
  let summary = "Buffer operation";
  let description = [{
    The buffer operation may buffer a `dc.value` or `dc.token` typed SSA value.
    In practice, this provides a mechanism to buffer data-side values in a
    control-sensitive manner.

    Example:
    ```mlir
    %value_out = dc.buffer [2] %value : !dc.value<i32, i1, i4>
    ```

    **Hardware/CIRCT context note**: buffers have no dialect-side notion of
    cycles/stages/implementation. It is up to the generating pass to interpret
    buffer semantics - some may want to add attributes to a single buffer, some
    may want to stagger `dc.buffer`s sequentially.
  }];

  let arguments = (ins
    ValueOrTokenType:$input,
    ConfinedAttr<I64Attr, [IntMinValue<1>]>:$size,
    OptionalAttr<ArrayAttr>:$initValues
  );
  let results = (outs ValueOrTokenType:$output);

  let assemblyFormat = "`[` $size `]` $input ($initValues^)? attr-dict `:` type($input)";
  let hasVerifier = 1;
  let builders = [OpBuilder<(
    ins "Value":$input, "size_t":$size), [{
        build($_builder, $_state, input.getType(), input, $_builder.getI64IntegerAttr(size), {});
    }]>];

  let extraClassDeclaration = [{
    // Returns the data type of this buffer, if any.
    std::optional<TypeRange> getInnerType() {
      if(auto type = getInput().getType().dyn_cast<ValueType>())
        return type.getInnerType();
      return std::nullopt;
    }

    // Returns the initial values of this buffer as a vector of int64's.
    FailureOr<SmallVector<int64_t>> getInitValueArray();
  }];
}

def JoinOp : DCOp<"join", [Commutative]> {
  let summary = "Synchronizes the incoming tokens with the outgoing token";
  let description = [{
    This operator synchronizes all incoming tokens. Synchronization implies applying
    join semantics in between all in- and output ports.

    Example:
    ```mlir
    %0 = dc.join %a, %b
    ```
  }];

  let arguments = (ins Variadic<TokenType>:$tokens);
  let results = (outs TokenType:$output);

  let assemblyFormat = "$tokens attr-dict";
  let hasFolder = 1;
}

def ForkOp : DCOp<"fork"> {
    let summary = "Splits the incoming token into multiple outgoing tokens";
    let description = [{
        This operator splits the incoming token into multiple outgoing tokens.

        Example:
        ```mlir
        %0, %1 = dc.fork [2] %a : !dc.token, !dc.token
        ```
    }];

    let arguments = (ins TokenType:$token);
    let results = (outs Variadic<TokenType>:$outputs);
    let hasCustomAssemblyFormat = 1;
    let hasCanonicalizer = 1;
    let hasFolder = 1;

    let builders = [OpBuilder<(
        ins "Value":$token, "size_t":$numOutputs), [{
            llvm::SmallVector<mlir::Type, 4> outputTypes(numOutputs, $_builder.getType<dc::TokenType>());
            build($_builder, $_state, outputTypes, token);
        }]>];
}

def BranchOp : DCOp<"branch"> {
    let summary = "Branch operation";
    let description = [{
        The incoming select token is propagated to the selected output based on
        the value of the condition.
    }];

    let arguments = (ins I1ValueType:$condition);
    let results = (outs TokenType:$trueToken, TokenType:$falseToken);

    let assemblyFormat = "$condition attr-dict";
}

def SelectOp : DCOp<"select"> {
    let summary = "Select operation";
    let description = [{
        An input token is selected based on the value of the incoming select
        signal, and propagated to the single output. Only the condition value,
        the selected input, and the output will be transacted.
    }];

    let arguments = (ins I1ValueType:$condition, TokenType:$trueToken, TokenType:$falseToken);
    let results = (outs TokenType:$output);
    let assemblyFormat = "$condition `,` $trueToken `,` $falseToken attr-dict";
    let hasCanonicalizer = 1;
}

def SinkOp : DCOp<"sink"> {
    let summary = "Sink operation";
    let description = [{
        The sink operation will always accept any incoming tokens, and 
        discard them.
    }];

    let arguments = (ins TokenType:$token);
    let results = (outs);
    let assemblyFormat = "$token attr-dict";
}

def SourceOp : DCOp<"source"> {
    let summary = "Source operation";
    let description = [{
        The source operation will always produce a token.
    }];

    let arguments = (ins);
    let results = (outs TokenType:$output);
    let assemblyFormat = "attr-dict";
    let builders = [OpBuilder<(ins), [{
            build($_builder, $_state, $_builder.getType<dc::TokenType>());
        }]>];
}

def PackOp : DCOp<"pack", [
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
    let summary = "Pack operation";
    let description = [{
        An operation which packs together a !dc.token value with some other
        value.

        Typically, a `dc.pack` op will be used to facilitate data-dependent
        control flow, wherein a `dc.value<i1>` is to be generated as a select
        signal for either a `dc.branch` or `dc.select` operation.
    }];

    let arguments = (ins TokenType:$token, AnyType:$input);
    let results = (outs ValueType:$output);
    let assemblyFormat = "$token `,` $input attr-dict `:` type($input)";
    let hasFolder = 1;
}

def UnpackOp : DCOp<"unpack", [
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
    let summary = "Unpack operation";
    let description = [{
        An operation which unpacks a !dc.value value into a !dc.token value
        and its constituent values.
    }];

    let arguments = (ins ValueType:$input);
    let results = (outs TokenType:$token, AnyType:$output);
    let assemblyFormat = "$input attr-dict `:` qualified(type($input))";
    let hasCanonicalizer = 1;
    let hasFolder = 1;
}

def MergeOp : DCOp<"merge"> {
  let summary = "Merge operation";
  let description = [{
    Select one of the incoming tokens and emits an output stating which token
    was selected. If multiple tokens are ready to transact at the same time,
    the tokens are selected with priority, from first to last (i.e. left to right
    in the IR). This property ensures deterministic behavior.
  }];

  let arguments = (ins TokenType:$first, TokenType:$second);
  let results = (outs I1ValueType:$output);
  let assemblyFormat = "$first `,` $second attr-dict";
}

def ToESIOp : DCOp<"to_esi", [
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let summary = "Convert a DC-typed value to an ESI-typed value";
  let description = [{
    Convert a `dc.token/dc.value` to an ESI channel.
  }];

  let arguments = (ins
    ValueOrTokenType:$input
  );
  let results = (outs ChannelType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def FromESIOp : DCOp<"from_esi", [
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
]> {
  let summary = "Convert an ESI-typed value to a DC-typed value";
  let description = [{
    Convert an ESI channel to a `dc.token/dc.value`.
  }];

  let arguments = (ins ChannelType:$input);
  let results = (outs ValueOrTokenType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // CIRCT_DIALECT_DC_OPS_TD
