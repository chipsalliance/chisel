//===- LLHDTypesImpl.td - LLHD data type definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic data type implementations for the LLHD dialect.
//
//===----------------------------------------------------------------------===//

// Base class for other typedefs. Provides dialact-specific defaults.
class LLHDType<string name> : TypeDef<LLHD_Dialect, name> { }

//===----------------------------------------------------------------------===//
// Type declarations
//===----------------------------------------------------------------------===//

// Declares the llhd::SigType in C++.
def SigTypeImpl : LLHDType<"Sig"> {
  let summary = "signal type";
  let description = [{
    Signals correspond directly to wires in a physical design, and are used to
    model propagation delays and timing. Signals are used to carry values across
    time steps in the LLHD execution model.
  }];

  let mnemonic = "sig";
  let parameters = (ins "::mlir::Type":$underlyingType);
  let assemblyFormat = "`<` $underlyingType `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$underlyingType), [{
      return $_get(underlyingType.getContext(), underlyingType);
    }]>
  ];
}

// Declares the llhd::PtrType in C++.
def PtrTypeImpl : LLHDType<"Ptr"> {
  let summary = "pointer type";
  let description = [{
    Represents a pointer to a memory location holding a value of its element
    type. May be used to load and store data in distinct memory slots.
  }];

  let mnemonic = "ptr";
  let parameters = (ins "::mlir::Type":$underlyingType);
  let assemblyFormat = "`<` $underlyingType `>`";

  let builders = [
    AttrBuilderWithInferredContext<(ins "::mlir::Type":$underlyingType), [{
      return $_get(underlyingType.getContext(), underlyingType);
    }]>
  ];
}

// Declares the llhd::TimeType in C++.
def TimeTypeImpl : LLHDType<"Time"> {
  let summary = "time type";
  let description = [{
    Represents a simulation time value as a combination of a real time value in
    seconds (or any smaller SI time unit), a delta value representing
    infinitesimal time steps, and an epsilon value representing an absolute time
    slot within a delta step (used to model SystemVerilog scheduling regions).
  }];

  let mnemonic = "time";
}

//===----------------------------------------------------------------------===//
// Attribute declarations
//===----------------------------------------------------------------------===//

// Declares the llhd::TimeAttr in C++.
def LLHD_TimeAttr : AttrDef<LLHD_Dialect, "Time", []> {
  let summary = "time attribute";
  let description = [{
    Represents a value of the LLHD time type.

    Example: `#llhd.time<0ns, 1d, 0e>`
  }];

  let mnemonic = "time";
  let storageType= [{ llhd::TimeAttr }];
  let genVerifyDecl = 1;

  let hasCustomAssemblyFormat = 1;

  let parameters = (ins AttributeSelfTypeParameter<"", "llhd::TimeType">:$type,
                        "unsigned":$time,
                        StringRefParameter<"SI time unit">:$timeUnit,
                        "unsigned":$delta,
                        "unsigned":$epsilon);

  let skipDefaultBuilders = 1;
  let builders = [ AttrBuilder<(ins "unsigned":$time,
                                    "llvm::StringRef":$timeUnit,
                                    "unsigned":$delta,
                                    "unsigned":$epsilon), [{
    return $_get($_ctxt, llhd::TimeType::get($_ctxt),
      time, timeUnit, delta, epsilon);
  }]>];
}
