//===- CHIRRTL.td - CHIRRTL dialect definition -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the top level file for the CHIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CHIRRTL_TD
#define CHIRRTL_TD

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "circt/Dialect/FIRRTL/FIRRTLDialect.td"
include "circt/Dialect/FIRRTL/FIRRTLTypes.td"
include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.td"
include "circt/Dialect/FIRRTL/FIRRTLAttributes.td"
include "circt/Dialect/FIRRTL/FIRRTLEnums.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/SymbolInterfaces.td"

def CHIRRTLDialect : Dialect {
  let name = "chirrtl";
  let cppNamespace = "::circt::chirrtl";

  let summary = "Types and operations for the chirrtl dialect";
  let description = [{
    This dialect defines the `chirrtl` dialect, which contains high-level
    memory defintions which can be lowered to FIRRTL.
  }];
  let dependentDialects = ["circt::firrtl::FIRRTLDialect"];

  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// CHIRRTL Types
//===----------------------------------------------------------------------===//

def CMemoryType : TypeDef<CHIRRTLDialect, "CMemory"> {
  let mnemonic = "cmemory";
  let summary = "a behavioral memory";
  let description = [{
    Syntax:
    ```
    cmemory-type ::= `cmemory` `<` element-type, element-count `>`
    ```

    The value of a `cmemory` type represents a behavioral memory with unknown
    ports.  This is produced by `combmem` and `seqmem` declarations and used by
    `memoryport` declarations to define memories and their ports. A CMemory is
    similar to a vector of passive element types.

    Examples:

    ```mlir
    !chirrtl.cmemory<uint<32>, 16>
    !chirrtl.cmemory<bundle<a : uint<1>>, 16>
    ```
  }];

  let parameters = (ins "firrtl::FIRRTLBaseType":$elementType,
                        "uint64_t":$numElements);

  let builders = [
    TypeBuilderWithInferredContext<(ins "firrtl::FIRRTLBaseType":$elementType,
                                        "uint64_t":$numElements), [{
      return $_get(elementType.getContext(), elementType, numElements);
    }]>
  ];

  let genVerifyDecl = 1;
  let hasCustomAssemblyFormat = 1;
}

def CMemoryPortType : TypeDef<CHIRRTLDialect, "CMemoryPort"> {
  let mnemonic = "cmemoryport";
  let summary = "a behavioral memory port";
  let description = [{
    Syntax:
    ```
    cmemoryport-type ::= `cmemoryport`
    ```

    The value of a `cmemoryport` type represents a port which has been
    declared on a `cmemory`.  This value is used to set the memory port access
    conditions.
  }];
}

//===----------------------------------------------------------------------===//
// CHIRRTL Operations
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect.
class CHIRRTLOp<string mnemonic, list<Trait> traits = []> :
    Op<CHIRRTLDialect, mnemonic, traits>;

def CombMemOp : CHIRRTLOp<"combmem", [HasCustomSSAName, DeclareOpInterfaceMethods<InnerSymbol,["getTargetResultIndex"]>, FNamableOp]> {
  let summary = "Define a new combinational memory";
  let description = [{
    Define a new behavioral combinational memory. Combinational memories have a
    write latency of 1 and a read latency of 0.
  }];
  let arguments = (ins StrAttr:$name, NameKindAttr:$nameKind,
                       AnnotationArrayAttr:$annotations,
                       OptionalAttr<InnerSymAttr>:$inner_sym,
                       OptionalAttr<MemoryInitAttr>:$init);
  let results = (outs CMemoryType:$result);
  let assemblyFormat = [{(`sym` $inner_sym^)? `` custom<NameKind>($nameKind)
                         `` custom<CombMemOp>(attr-dict) `:` qualified(type($result))}];
  let builders = [
    OpBuilder<(ins "firrtl::FIRRTLBaseType":$elementType, "uint64_t":$numElements,
                   "mlir::StringRef":$name, "firrtl::NameKindEnum":$nameKind,
                   "ArrayAttr":$annotations,
                   CArg<"StringAttr", "StringAttr()">:$innerSym,
                   CArg<"firrtl::MemoryInitAttr", "firrtl::MemoryInitAttr{}">:$init)>
  ];
}

def SeqMemOp : CHIRRTLOp<"seqmem", [HasCustomSSAName, DeclareOpInterfaceMethods<InnerSymbol,["getTargetResultIndex"]>, FNamableOp]> {
  let summary = "Define a new sequential memory";
  let description = [{
    Define a new behavioral sequential memory.  Sequential memories have a
    write latency and a read latency of 1.
  }];
  let arguments = (ins RUWAttr:$ruw, StrAttr:$name, NameKindAttr:$nameKind,
                       AnnotationArrayAttr:$annotations,
                       OptionalAttr<InnerSymAttr>:$inner_sym,
                       OptionalAttr<MemoryInitAttr>:$init);
  let results = (outs CMemoryType:$result);
  let assemblyFormat = [{(`sym` $inner_sym^)? `` custom<NameKind>($nameKind) $ruw
                         custom<SeqMemOp>(attr-dict) `:` qualified(type($result))}];
  let builders = [
    OpBuilder<(ins "firrtl::FIRRTLBaseType":$elementType, "uint64_t":$numElements,
                   "RUWAttr":$ruw, "mlir::StringRef":$name,
                   "firrtl::NameKindEnum":$nameKind, "ArrayAttr":$annotations,
                   CArg<"StringAttr", "StringAttr()">:$innerSym,
                   CArg<"firrtl::MemoryInitAttr", "firrtl::MemoryInitAttr{}">:$init)>
  ];
}

def MemoryPortOp : CHIRRTLOp<"memoryport", [InferTypeOpInterface,
      HasCustomSSAName]> {
  let summary = "Declares a memory port on a memory";

  let summary = "Defines a memory port on CHIRRTL memory";
  let description = [{
    This operation defines a new memory port on a `seqmem` or `combmem`CHISEL.
    `data` is the data returned from the memory port.

    The memory port requires an access point, which sets the enable condition
    of the port, the clock, and the address.  This is done by passing the the
    `port` argument to a `chirrtl.memoryport.access operation`.

  }];

  let arguments = (ins CMemoryType:$memory, MemDirAttr:$direction,
                    StrAttr:$name, AnnotationArrayAttr:$annotations);

  let results = (outs FIRRTLBaseType:$data, CMemoryPortType:$port);

  let assemblyFormat = [{
    $direction $memory `` custom<MemoryPortOp>(attr-dict) `:`
       functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Type":$dataType, "::mlir::Value":$memory,
                   "MemDirAttr":$direction, CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations)>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Get the assocated access op.
    MemoryPortAccessOp getAccess();

    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];
}

def MemoryPortAccessOp : CHIRRTLOp<"memoryport.access"> {
  let summary = "Enables a memory port";

  let description = [{
    This operation is used to conditionally enable a memory port, and associate
    it with a `clock` and `index`.  The memory port will be actuve on the
    positive edge of the clock. The index is the address of the memory
    accessed. See the FIRRTL rational for more information about why this
    operation exists.
  }];

  let arguments = (ins CMemoryPortType:$port, IntType:$index,
                   ClockType:$clock);

  let assemblyFormat = [{
    $port `[` $index `]` `,` $clock attr-dict `:` qualified(type(operands))
  }];
}

def MemoryDebugPortOp : CHIRRTLOp<"debugport", [InferTypeOpInterface,
      HasCustomSSAName]> {
  let summary = "Declares a memory debug port on a memory";

  let summary = "Defines a debug memory port on CHIRRTL memory";
  let description = [{
    This operation defines a new debug memory port on a `combmem`CHISEL.
    `data` is the data returned from the memory port.
  }];

  let arguments = (ins CMemoryType:$memory,
                    StrAttr:$name, AnnotationArrayAttr:$annotations);

  let results = (outs RefType:$data);

  let assemblyFormat = [{
    $memory custom<MemoryDebugPortOp>(attr-dict) `:`
       functional-type(operands, results)
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Type":$dataType, "::mlir::Value":$memory,
                   CArg<"StringRef", "{}">:$name,
                   CArg<"ArrayRef<Attribute>","{}">:$annotations)>
  ];

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          std::optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::OpaqueProperties properties,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];
}
#endif // CHIRRTL_TD
