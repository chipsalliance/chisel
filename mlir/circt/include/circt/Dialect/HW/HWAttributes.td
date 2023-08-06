//===- HWAttributes.td - Attributes for HW dialect ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines HW dialect specific attributes.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWATTRIBUTES_TD
#define CIRCT_DIALECT_HW_HWATTRIBUTES_TD

include "circt/Dialect/HW/HWDialect.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

/// An attribute to indicate the output file an operation should be emitted to.
def OutputFileAttr : AttrDef<HWDialect, "OutputFile"> {
  let summary = "Output file attribute";
  let description = [{
    This attribute represents an output file for something which will be
    printed. The `filename` string is the file to be output to. If `filename`
    ends in a `/` it is considered an output directory.

    When ExportVerilog runs, one of the files produced is a list of all other
    files which are produced. The flag `excludeFromFileList` controls if this
    file should be included in this list. If any `OutputFileAttr` referring to
    the same file sets this to `true`, it will be included in the file list.
    This option defaults to `false`.

    For each file emitted by the verilog emitter, certain prelude output will
    be included before the main content. The flag `includeReplicatedOps` can
    be used to disable the addition of the prelude text. All `OutputFileAttr`s
    referring to the same file must use a consistent setting for this value.
    This option defaults to `true`.

    Examples:
    ```mlir
      #hw.ouput_file<"/home/tester/t.sv">
      #hw.ouput_file<"t.sv", excludeFromFileList, includeReplicatedOps>
    ```
  }];
  let mnemonic = "output_file";
  let parameters = (ins "::mlir::StringAttr":$filename,
                        "::mlir::BoolAttr":$excludeFromFilelist,
                        "::mlir::BoolAttr":$includeReplicatedOps);
  let builders = [
    AttrBuilderWithInferredContext<(ins
                                   "::mlir::StringAttr":$filename,
                                   "::mlir::BoolAttr":$excludeFromFileList,
                                   "::mlir::BoolAttr":$includeReplicatedOps), [{
      return get(filename.getContext(), filename, excludeFromFileList,
                 includeReplicatedOps);
    }]>,
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Get an OutputFileAttr from a string filename, canonicalizing the
    /// filename.
    static OutputFileAttr getFromFilename(::mlir::MLIRContext *context,
                                          const ::mlir::Twine &filename,
                                          bool excludeFromFileList = false,
                                          bool includeReplicatedOps = false);

    /// Get an OutputFileAttr from a string filename, resolving it relative to
    /// `directory`. If `filename` is an absolute path, the given `directory`
    /// will not be used.
    static OutputFileAttr getFromDirectoryAndFilename(
                                          ::mlir::MLIRContext *context,
                                          const ::mlir::Twine &directory,
                                          const ::mlir::Twine &filename,
                                          bool excludeFromFileList = false,
                                          bool includeReplicatedOps = false);

    /// Get an OutputFileAttr from a string directory name. The name will have
    /// a trailing `/` added if it is not there, ensuring that this will be
    /// an output directory.
    static OutputFileAttr getAsDirectory(::mlir::MLIRContext *context,
                                         const ::mlir::Twine &directory,
                                         bool excludeFromFileList = false,
                                         bool includeReplicatedOps = false);

    /// Returns true if this a directory.
    bool isDirectory();
  }];
}

// An attribute to indicate which filelist an operation's file should be
// included in.
def FileListAttr : AttrDef<HWDialect, "FileList"> {
  let summary = "Ouput filelist attribute";
  let description = [{
    This attribute represents an output filelist for something which will be
    printed. The `filename` string is the file which the filename of the
    operation to be output to.

    When ExportVerilog runs, some of the files produced are lists of other files
    which are produced.  Each filelist exported contains entities' output file
    with `FileListAttr` marked.


    Examples:
    ```mlir
      #hw.ouput_filelist<"/home/tester/t.F">
      #hw.ouput_filelist<"t.f">
    ```
  }];
  let mnemonic = "output_filelist";
  let parameters = (ins "::mlir::StringAttr":$filename);
  let builders = [
    AttrBuilderWithInferredContext<(ins
                                   "::mlir::StringAttr":$filename), [{
      return get(filename.getContext(), filename);
    }]>,
  ];

  let assemblyFormat = "`<` $filename `>`";

  let extraClassDeclaration = [{
    /// Get an OutputFileAttr from a string filename, canonicalizing the
    /// filename.
    static FileListAttr getFromFilename(::mlir::MLIRContext *context,
                                          const ::mlir::Twine &filename);
  }];

}

def ParamDeclAttr : AttrDef<HWDialect, "ParamDecl", [TypedAttrInterface]> {
  let summary = "Module or instance parameter definition";
  let description = [{
    An attribute describing a module parameter, or instance parameter
    specification.
  }];

  /// The value of the attribute - in a module, this is the default
  /// value (and may be missing).  In an instance, this is a required field that
  /// specifies the value being passed.  The verilog emitter omits printing the
  /// parameter for an instance when the applied value and the default value are
  /// the same.
  let parameters = (ins "::mlir::StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type,
                        "::mlir::Attribute":$value);
  let mnemonic = "param.decl";

  let hasCustomAssemblyFormat = 1;

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                        "::mlir::Type":$type),
      "auto *context = type.getContext();\n"
      "return $_get(context, name, type, Attribute());">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                        "::mlir::Type":$type),
      "return get(StringAttr::get(type.getContext(), name), type);">,

    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                        "::mlir::TypedAttr":$value),
      "auto *context = value.getContext();\n"
      "return $_get(context, name, value.getType(), value);">,
    AttrBuilderWithInferredContext<(ins "::mlir::StringRef":$name,
                                        "::mlir::TypedAttr":$value),
      "return get(StringAttr::get(value.getContext(), name), value);">
  ];

  let extraClassDeclaration = [{
    static ParamDeclAttr getWithName(ParamDeclAttr param,
                                     ::mlir::StringAttr name) {
      return get(param.getContext(), name, param.getType(), param.getValue());
    }
  }];
}

/// An array of ParamDeclAttr's that may or may not have a 'value' specified,
/// to be used on hw.module or hw.instance.  The hw.instance verifier further
/// ensures that all the values are specified.
def ParamDeclArrayAttr
  : TypedArrayAttrBase<ParamDeclAttr, "parameter array attribute">;

/// This attribute models a reference to a named parameter within a module body.
/// The type of the ParamDeclRefAttr must always be the same as the type of the
/// parameter being referenced.
def ParamDeclRefAttr : AttrDef<HWDialect, "ParamDeclRef",
                               [TypedAttrInterface]> {
  let summary = "Is a reference to a parameter value.";
  let parameters = (ins "::mlir::StringAttr":$name,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "param.decl.ref";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::StringAttr":$name,
                                        "::mlir::Type":$type), [{
      return get(name.getContext(), name, type);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
}

def ParamVerbatimAttr : AttrDef<HWDialect, "ParamVerbatim",
                                [TypedAttrInterface]> {
  let summary =
    "Represents text to emit directly to SystemVerilog for a parameter";
  let parameters = (ins "::mlir::StringAttr":$value,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "param.verbatim";

  let hasCustomAssemblyFormat = 1;
}


/// Parameter Expression Opcodes.
let cppNamespace = "circt::hw" in {

/// Fully Associative Expression Opcodes.
def PEO_Add : I32EnumAttrCase<"Add", 0, "add">;
def PEO_Mul : I32EnumAttrCase<"Mul", 1, "mul">;
def PEO_And : I32EnumAttrCase<"And", 2, "and">;
def PEO_Or  : I32EnumAttrCase<"Or",  3, "or">;
def PEO_Xor : I32EnumAttrCase<"Xor", 4, "xor">;

// Binary Expression Opcodes.
def PEO_Shl  : I32EnumAttrCase<"Shl" , 5, "shl">;
def PEO_ShrU : I32EnumAttrCase<"ShrU", 6, "shru">;
def PEO_ShrS : I32EnumAttrCase<"ShrS", 7, "shrs">;
def PEO_DivU : I32EnumAttrCase<"DivU", 8, "divu">;
def PEO_DivS : I32EnumAttrCase<"DivS", 9, "divs">;
def PEO_ModU : I32EnumAttrCase<"ModU",10, "modu">;
def PEO_ModS : I32EnumAttrCase<"ModS",11, "mods">;

// Unary Expression Opcodes.
def PEO_CLog2 : I32EnumAttrCase<"CLog2", 12, "clog2">;

// String manipulation Opcodes.
def PEO_StrConcat : I32EnumAttrCase<"StrConcat", 13, "str.concat">;

def PEOAttr  : I32EnumAttr<"PEO", "Parameter Expression Opcode",
                           [PEO_Add, PEO_Mul, PEO_And, PEO_Or, PEO_Xor,
                            PEO_Shl, PEO_ShrU, PEO_ShrS,
                            PEO_DivU, PEO_DivS, PEO_ModU, PEO_ModS,
                            PEO_CLog2, PEO_StrConcat]>;
}

def ParamExprAttr : AttrDef<HWDialect, "ParamExpr", [TypedAttrInterface]> {
  let summary = "Parameter expression combining operands";
  let parameters = (ins "PEO":$opcode,
                        ArrayRefParameter<"::mlir::TypedAttr">:$operands,
                        AttributeSelfTypeParameter<"">:$type);
  let mnemonic = "param.expr";

  // Force all clients to go through our building logic so we can canonicalize
  // during building.
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    /// Build a parameter expression.  This automatically canonicalizes and
    /// folds, so it may not necessarily return a ParamExprAttr.
    static mlir::TypedAttr get(PEO opcode,
                               mlir::ArrayRef<mlir::TypedAttr> operands);

    /// Build a binary parameter expression for convenience.
    static mlir::TypedAttr get(PEO opcode, mlir::TypedAttr lhs,
                                           mlir::TypedAttr rhs) {
      mlir::TypedAttr operands[] = { lhs, rhs };
      return get(opcode, operands);
    }
  }];

  let hasCustomAssemblyFormat = 1;
}

// An attribute to indicate an enumeration value.
def EnumFieldAttr : AttrDef<HWDialect, "EnumField"> {
  let summary = "Enumeration field attribute";
  let description = [{
    This attribute represents a field of an enumeration.

    Examples:
    ```mlir
      #hw.enum.value<A, !hw.enum<A, B, C>>
    ```
  }];
  let mnemonic = "enum.field";
  let parameters = (ins "::mlir::StringAttr":$field, "::mlir::TypeAttr":$type);

  // Force all clients to go through our custom builder so we can check
  // whether the requested enum value is part of the provided enum type.
  let skipDefaultBuilders = 1;
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    /// Builds a new EnumFieldAttr of the provided value.
    /// This will fail if the value is not a member of the provided enum type.
    static EnumFieldAttr get(::mlir::Location loc, ::mlir::StringAttr value, mlir::Type type);
  }];
}

#endif // CIRCT_DIALECT_HW_HWATTRIBUTES_TD
