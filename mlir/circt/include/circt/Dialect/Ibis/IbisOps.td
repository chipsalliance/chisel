//===- IbisOps.td - Definition of Ibis dialect operations -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_IBISOPS_TD
#define CIRCT_DIALECT_IBIS_IBISOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "circt/Dialect/HW/HWOpInterfaces.td"
include "circt/Dialect/Ibis/IbisInterfaces.td"

def HasCustomSSAName :
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>;

def ClassOp : IbisOp<"class", [
    IsolatedFromAbove, RegionKindInterface,
    Symbol, SymbolTable, SingleBlock,
    NoTerminator, NoRegionArguments,
    HasParent<"mlir::ModuleOp">]> {

  let summary = "Ibis class";
  let description = [{
    Ibis has the notion of a class which can contain methods and member
    variables.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph; }
  }];
}

def InstanceOp : IbisOp<"instance", [
    Symbol,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Ibis class instance";
  let description = [{
    Instantiates an Ibis class.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, FlatSymbolRefAttr:$className);
  let assemblyFormat = [{
    $sym_name `,` $className attr-dict
  }];

  let extraClassDeclaration = [{
    // Return the class this instance is instantiating.
    ClassOp getClass(SymbolTableCollection *symbolTable = nullptr) {
      auto mod = getOperation()->getParentOfType<mlir::ModuleOp>();
      if(symbolTable)
        return symbolTable->lookupSymbolIn<ClassOp>(mod, getClassNameAttr());

      return mod.lookupSymbol<ClassOp>(getClassNameAttr());
    }
  }];
}

def MethodOp : IbisOp<"method", [
      IsolatedFromAbove, RegionKindInterface,
      Symbol, FunctionOpInterface,
      DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
      SingleBlockImplicitTerminator<"ReturnOp">,
      HasParent<"ClassOp">]> {

  let summary = "Ibis method";
  let description = [{
    Ibis functions are a lot like software functions: a list of named arguments
    and one unnamed return value.

    Can only live inside of classes.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       ArrayAttr:$argNames,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region SizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // Implement RegionKindInterface.
    static RegionKind getRegionKind(unsigned index) { return RegionKind::Graph;}

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}

def ReturnOp : IbisOp<"return", [
      Pure, ReturnLike, Terminator, HasParent<"MethodOp">]> {
  let summary = "Ibis function terminator";

  let arguments = (ins Optional<AnyType>:$retValue);
  let assemblyFormat = "($retValue^)? attr-dict (`:` type($retValue)^)?";
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins)>,
  ];
}

def CallOp : IbisOp<"call", [CallOpInterface]> {
  let summary = "Ibis method call";
  let description = [{
    Dispatch a call to an Ibis method.
  }];

  let arguments = (ins SymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    MutableOperandRange getArgOperandsMutable() {
      return getOperandsMutable();
    }

    /// Get the argument operands to the called method.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }

    /// Set the callee for this operation.
    void setCalleeFromCallable(mlir::CallInterfaceCallable callee) {
      (*this)->setAttr(getCalleeAttrName(), callee.get<mlir::SymbolRefAttr>());
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}


// ===---------------------------------------------------------------------===//
// Low-level Ibis operations
// ===---------------------------------------------------------------------===//

def ContainerOp : IbisOp<"container", [
  IsolatedFromAbove,
  Symbol, SymbolTable, SingleBlock,
  NoTerminator, NoRegionArguments,
  HasParent<"ClassOp">
]> {
  let summary = "Ibis container";
  let description = [{
    An ibis container describes a collection of logic nested within an Ibis class.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name attr-dict-with-keyword $body
  }];
}

class PortLikeOp<string mnemonic, list<Trait> traits = []> :
    IbisOp<mnemonic, !listconcat(traits, [
      PortOpInterface,
      HasParent<"ClassOp">
  ])> {
  let arguments = (ins SymbolNameAttr:$sym_name, TypeAttrOf<AnyType>:$type);
  let assemblyFormat = [{
    $sym_name `:` $type attr-dict
  }];
}

def InputPortOp : PortLikeOp<"port.input"> {
  let summary = "Ibis input port";
}

def OutputPortOp : PortLikeOp<"port.output"> {
  let summary = "Ibis output port";
}

def PortReadOp : IbisOp<"port.read", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Ibis port read";
  let description = [{
    Read the value of a local port (i.e. a port of 'this' class).
  }];

  let arguments = (ins FlatSymbolRefAttr:$sym_name);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $sym_name attr-dict `:` type($output)
  }];
}

def PortWriteOp : IbisOp<"port.write", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Ibis port write";
  let description = [{
    Write a value to a local port (i.e. a port of 'this' class).
  }];

  let arguments = (ins FlatSymbolRefAttr:$sym_name, AnyType:$input);
  let assemblyFormat = [{
    $sym_name `(` $input `)` attr-dict `:` type($input)
  }];
}

def InstanceReadOp : IbisOp<"instance.port.read", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Ibis instance read";
  let description = [{
    Read the value of an instance port.
  }];

  let arguments = (ins SymbolRefAttr:$sym_name);
  let results = (outs AnyType:$output);
  let assemblyFormat = [{
    $sym_name attr-dict `:` type($output)
  }];
}

def InstanceWriteOp : IbisOp<"instance.port.write", [
      DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
  let summary = "Ibis instance write";
  let description = [{
    Write a value to an instance port.
  }];

  let arguments = (ins SymbolRefAttr:$sym_name, AnyType:$input);
  let assemblyFormat = [{
    $sym_name `(` $input `)` attr-dict `:` type($input)
  }];
}

#endif // CIRCT_DIALECT_IBIS_IBISOPS_TD
