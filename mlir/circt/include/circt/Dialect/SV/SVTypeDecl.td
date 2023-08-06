//===- SVTypeDecl.td - SV type declaration ops -------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for SystemVerilog type declartions.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWOpInterfaces.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Interface operations
//===----------------------------------------------------------------------===//

def InterfaceOp : SVOp<"interface",
    [SingleBlock, NoTerminator, Symbol, SymbolTable]> {
  let summary = "Operation to define a SystemVerilog interface";

  let description = [{
    This operation defines a named interface. Its name is a symbol that can
    be looked up when declared inside a SymbolTable operation. This operation is
    also a SymbolTable itself, so the symbols in its region can be looked up.

    Example:

    ```mlir
    sv.interface @myinterface {
      sv.interface.signal @data : i32
      sv.interface.modport @input_port (input @data)
      sv.interface.modport @output_port (output @data)
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$comment
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = "$sym_name attr-dict-with-keyword $body";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$sym_name,
               CArg<"std::function<void()>", "{}">:$body)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }

    ModportType getModportType(::llvm::StringRef);
    InterfaceType getInterfaceType() {
      return InterfaceType::get(getContext(),
        ::mlir::SymbolRefAttr::get(getContext(), getSymName()));
    }
    Type getSignalType(StringRef signalName);

    // Get the interface's symbolic name as StringAttr.
    StringAttr getNameAttr() {
      return (*this)->getAttrOfType<StringAttr>(
        ::mlir::SymbolTable::getSymbolAttrName());
    }

    // Get the interface's symbolic name.
    StringRef getName() {
      return getNameAttr().getValue();
    }
  }];
}

def SignalTypeAttr : TypeAttrBase<"::mlir::Type", "Any SV/HW type"> {
  let predicate = And<[
      CPred<"$_self.isa<::mlir::TypeAttr>()">,
      CPred<"::circt::hw::getBitWidth(" #
              "$_self.cast<::mlir::TypeAttr>().getValue()) != -1">
  ]>;
}

def InterfaceSignalOp : SVOp<"interface.signal",
    [Symbol, HasParent<"InterfaceOp">]> {
  let summary = "Operation to define a SystemVerilog signal for interfaces";

  let description = [{
    This operation defines a named signal within an interface. Its type is
    specified in an attribute, and currently supports IntegerTypes.

    Example:

    ```mlir
    sv.interface.signal @data : i32
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    SignalTypeAttr:$type
  );

  let assemblyFormat = "attr-dict $sym_name `:` $type";

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name, "::mlir::Type":$type)>
  ];
}

def ModportDirectionInput : I32EnumAttrCase<"input", 0>;
def ModportDirectionOutput : I32EnumAttrCase<"output", 1>;
def ModportDirectionInOut : I32EnumAttrCase<"inout", 2>;

def ModportDirection : I32EnumAttr<"ModportDirection",
  "Defines direction in a modport",
  [ModportDirectionInput, ModportDirectionOutput, ModportDirectionInOut]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::circt::sv";
}

def ModportDirectionAttr : EnumAttr<SVDialect, ModportDirection,
  "modport_direction">;

class SV_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<SVDialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

def ModportStruct : SV_Attr<"ModportStruct", "mod_port"> {
  let parameters = (ins
                    "::circt::sv::ModportDirectionAttr":$direction,
                    "mlir::FlatSymbolRefAttr":$signal);

  let assemblyFormat = "`<` struct(params) `>`";

  let builders = [
    AttrOrTypeBuilder<(ins "ModportDirection":$dir, "mlir::FlatSymbolRefAttr":$sig)>
  ];
}

def ModportStructArrayAttr : TypedArrayAttrBase<ModportStruct,
  "array of modport structs">;

def InterfaceModportOp : SVOp<"interface.modport",
    [Symbol, HasParent<"InterfaceOp">]> {
  let summary = "Operation to define a SystemVerilog modport for interfaces";

  let description = [{
    This operation defines a named modport within an interface. Its name is a
    symbol that can be looked up inside its parent interface. There is an array
    of structs that contains two fields: an enum to indicate the direction of
    the signal in the modport, and a symbol reference to refer to the signal.

    Example:

    ```mlir
    sv.interface.modport @input_port (input @data)
    sv.interface.modport @output_port (output @data)
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    ModportStructArrayAttr:$ports
  );

  let assemblyFormat = "attr-dict $sym_name custom<ModportStructs>($ports)";

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$name,
                      "ArrayRef<::llvm::StringRef>":$inputs,
                      "ArrayRef<::llvm::StringRef>":$outputs)>
  ];
}

//===----------------------------------------------------------------------===//
// Types for integration in to type-based dialects.
//===----------------------------------------------------------------------===//

def InterfaceType : SVType<"Interface"> {
  let summary = "SystemVerilog interface type pointing to an InterfaceOp";
  let description = [{
    A MLIR type for the SV dialect's `InterfaceOp` to allow instances in any
    dialect with an open type system. Points at the InterfaceOp which defines
    the SystemVerilog interface.
  }];

  let mnemonic = "interface";
  let parameters = (ins "::mlir::FlatSymbolRefAttr":$interface);
  let assemblyFormat = "`<` $interface `>`";
}

def ModportType : SVType<"Modport"> {
  let summary = "SystemVerilog type pointing to an InterfaceModportOp";
  let description = [{
    A MLIR type for the SV dialect's `InterfaceModportOp` to allow
    interactions with any open type system dialect. Points at the
    InterfaceModportOp which defines the SystemVerilog interface's modport.
  }];

  let mnemonic = "modport";
  let parameters = (ins "::mlir::SymbolRefAttr":$modport);
  let assemblyFormat = "`<` $modport `>`";
}

//===----------------------------------------------------------------------===//
// Other operations
//===----------------------------------------------------------------------===//

def InterfaceInstanceOp : SVOp<"interface.instance", [
      HasCustomSSAName,
      DeclareOpInterfaceMethods<InnerSymbol, ["getTargetResultIndex"]>
    ]> {
  let summary = "Instantiate an interface";
  let description = [{
    Use this to declare an instance of an interface:
    ```
    %iface = sv.interface.instance : !sv.interface<@handshake_example>
    ```
  }];

  let arguments = (ins StrAttr:$name,
                       OptionalAttr<InnerSymAttr>:$inner_sym);
  let results = (outs InterfaceType : $result);

  let assemblyFormat = [{
    (`sym` $inner_sym^)? `` custom<ImplicitSSAName>($name) attr-dict
    `:` qualified(type($result))
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Type":$result), [{
      return build($_builder, $_state, result, std::nullopt);
    }]>
  ];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    InterfaceType getInterfaceType() { return getType().cast<InterfaceType>(); }

    /// Lookup the interface that this is an instance of.  This returns null on
    /// invalid IR.
    Operation *getReferencedInterface(const hw::HWSymbolCache *cache = nullptr);


    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    /// An InstanceOp may optionally define a symbol.
    bool isOptionalSymbol() { return true; }

  }];
}

def GetModportOp: SVOp<"modport.get", [Pure]> {
  let summary = "Get a modport out of an interface instance";
  let description = [{
    Use this to extract a modport view to an instantiated interface. For
    example, to get the 'dataflow_in' modport on the 'handshake_example'
    interface:
    ```
    %ifaceModport = sv.modport.get @dataflow_in %iface :
      !sv.interface<@handshake_example> ->
      !sv.modport<@handshake_example::@dataflow_in>
    ```
  }];

  let arguments = (ins InterfaceType:$iface, FlatSymbolRefAttr:$field);
  let results = (outs ModportType:$result);

  let assemblyFormat =
    "$iface $field attr-dict `:` qualified(type($iface)) `->` qualified(type($result))";

  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "Value":$value, "::llvm::StringRef":$field)>
  ];

  let extraClassDeclaration = [{
    /// Lookup the op for the modport declaration.  This returns null on invalid
    /// IR.
    InterfaceModportOp getReferencedDecl(const hw::HWSymbolCache &cache);
  }];
}

def AssignInterfaceSignalOp : SVOp<"interface.signal.assign", []> {
  let summary = "Assign an interfaces signal to some other signal.";
  let description = [{
    Use this to continuously assign a signal inside an interface to a
    value or other signal.

    ```
      sv.interface.signal.assign %iface(@handshake_example::@data)
        = %zero32 : i32
    ```

    Would result in the following SystemVerilog:
    ```
      assign iface.data = zero32;
    ```
  }];

  let arguments = (ins InterfaceType:$iface, FlatSymbolRefAttr:$signalName,
    AnyType:$rhs);

  let assemblyFormat = [{
    $iface `(` custom<IfaceTypeAndSignal>(type($iface), $signalName) `)`
        `=` $rhs attr-dict `:` qualified(type($rhs))
  }];

  let hasVerifier = 1;
}

def ReadInterfaceSignalOp : SVOp<"interface.signal.read", [Pure]> {
  let summary = "Access the data in an interface's signal.";
  let description = [{
    This is an expression to access a signal inside of an interface.

    ```
      %ifaceData = sv.interface.signal.read %iface
          (@handshake_example::@data) : i32
    ```

    Could result in the following SystemVerilog:
    ```
      wire [31:0] ifaceData = iface.data;
    ```
  }];

  let arguments = (ins InterfaceType:$iface, FlatSymbolRefAttr:$signalName);
  let results = (outs AnyType:$signalData);

  let assemblyFormat = [{
    $iface `(` custom<IfaceTypeAndSignal>(type($iface), $signalName) `)`
        attr-dict `:` qualified(type($signalData))
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$iface, "::llvm::StringRef":$signalName)>
  ];


  let extraClassDeclaration = [{
    /// Lookup the op for the signal declaration.  This returns null on invalid
    /// IR.
    InterfaceSignalOp getReferencedDecl(const hw::HWSymbolCache &cache);
  }];

  let hasVerifier = 1;
}
