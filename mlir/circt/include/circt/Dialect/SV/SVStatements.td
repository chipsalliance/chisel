//===- SVStatements.td - SV statements ops -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the ops for System Verilog statements.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

/// Ensure symbol is one of the hw module.* types
def isModuleSymbol : AttrConstraint<
  CPred<
    "hw::isAnyModule(::mlir::SymbolTable::lookupNearestSymbolFrom("
      "&$_op, $_self.cast<::mlir::FlatSymbolRefAttr>().getValue()))"
  >, "is module like">;

/// Ensure symbol is one of the sv macro.* types
def isMacroSymbol : AttrConstraint<
  CPred<
    "sv::isAnyMacro(::mlir::SymbolTable::lookupNearestSymbolFrom("
      "&$_op, $_self.cast<::mlir::FlatSymbolRefAttr>().getValue()))"
  >, "is macro like">;

//===----------------------------------------------------------------------===//
// Control flow like-operations
//===----------------------------------------------------------------------===//

def OrderedOutputOp : SVOp<"ordered", [SingleBlock, NoTerminator, NoRegionArguments,
                             NonProceduralOp]> {
  let summary = "a sub-graph region which guarantees to output statements in-order";

  let description = [{
    This operation groups operations into a region whose purpose is to force
    verilog emission to be statement-by-statement, in-order.  This allows
    side-effecting operations, or macro expansions which applie to subsequent
    operations to be properly sequenced.
    This operation is for non-procedural regions and its body is non-procedural.
  }];

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "$body attr-dict";
  let builders = [
    OpBuilder<(ins CArg<"std::function<void()>", "{}">:$body)>
  ];
}

def IfDefOp : SVOp<"ifdef", [SingleBlock, NoTerminator, NoRegionArguments,
                             NonProceduralOp]> {
  let summary = "'ifdef MACRO' block";

  let description = [{
    This operation is an #ifdef block, which has a "then" and "else" region.
    This operation is for non-procedural regions and its body is non-procedural.
  }];

  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
  let arguments = (ins MacroIdentAttr:$cond);
  let results = (outs);

  let hasCanonicalizeMethod = true;
  let assemblyFormat = "$cond $thenRegion (`else` $elseRegion^)? attr-dict";

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$cond,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>,
    OpBuilder<(ins "StringAttr":$cond,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>,
    OpBuilder<(ins "MacroIdentAttr":$cond,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>
  ];

  let extraClassDeclaration = [{
    /// Returns true if an else block exists.
    bool hasElse() { return !getElseRegion().empty(); }

    Block *getThenBlock() {
      assert(!getThenRegion().empty() && "Unexpected empty 'then' region.");
      return &getThenRegion().front();
    }

    Block *getElseBlock() {
      assert(hasElse() && "Empty 'else' region.");
      return &getElseRegion().front();
    }
  }];
}

def IfDefProceduralOp
  : SVOp<"ifdef.procedural", [SingleBlock, NoTerminator, NoRegionArguments,
                              ProceduralRegion, ProceduralOp]> {
  let summary = "'ifdef MACRO' block for procedural regions";

  let description = [{
    This operation is an #ifdef block, which has a "then" and "else" region.
    This operation is for procedural regions and its body is procedural.
  }];

  let hasCanonicalizeMethod = true;
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
  let arguments = (ins MacroIdentAttr:$cond);
  let results = (outs);

  let assemblyFormat = "$cond $thenRegion (`else` $elseRegion^)? attr-dict";

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "StringRef":$cond,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>,
    OpBuilder<(ins "StringAttr":$cond,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>,
    OpBuilder<(ins "MacroIdentAttr":$cond,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>
  ];

  let extraClassDeclaration = [{
    /// Returns true if an else block exists.
    bool hasElse() { return !getElseRegion().empty(); }

    Block *getThenBlock() {
      assert(!getThenRegion().empty() && "Unexpected empty 'then' region.");
      return &getThenRegion().front();
    }

    Block *getElseBlock() {
      assert(hasElse() && "Empty 'else' region.");
      return &getElseRegion().front();
    }
  }];
}

def IfOp : SVOp<"if", [SingleBlock, NoTerminator, NoRegionArguments,
                       ProceduralRegion, ProceduralOp]> {
  let summary = "'if (cond)' block";

  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);
  let arguments = (ins I1:$cond);
  let results = (outs);

  let hasCanonicalizeMethod = true;
  let assemblyFormat = "$cond $thenRegion (`else` $elseRegion^)? attr-dict";

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$cond,
                      CArg<"std::function<void()>", "{}">:$thenCtor,
                      CArg<"std::function<void()>", "{}">:$elseCtor)>
  ];

  let extraClassDeclaration = [{
    /// Returns true if an else block exists.
    bool hasElse() { return !getElseRegion().empty(); }

    Block *getThenBlock() {
      assert(!getThenRegion().empty() && "Unexpected empty 'then' region.");
      return &getThenRegion().front();
    }

    Block *getElseBlock() {
      assert(hasElse() && "Empty 'else' region.");
      return &getElseRegion().front();
    }
  }];
}

// Edge behavior enum for always block.  See SV Spec 9.4.2.

/// AtPosEdge triggers on a rise from 0 to 1/X/Z, or X/Z to 1.
def AtPosEdge: I32EnumAttrCase<"AtPosEdge", 0, "posedge">;
/// AtNegEdge triggers on a drop from 1 to 0/X/Z, or X/Z to 0.
def AtNegEdge: I32EnumAttrCase<"AtNegEdge", 1, "negedge">;
/// AtEdge(v) is syntactic sugar for "AtPosEdge(v) or AtNegEdge(v)".
def AtEdge   : I32EnumAttrCase<"AtEdge", 2, "edge">;

def EventControlAttr : I32EnumAttr<"EventControl", "edge control trigger",
                                   [AtPosEdge, AtNegEdge, AtEdge]> {
  let cppNamespace = "circt::sv";
}


def AlwaysOp : SVOp<"always", [SingleBlock, NoTerminator, NoRegionArguments,
                               RecursiveMemoryEffects, RecursivelySpeculatable,
                               ProceduralRegion, NonProceduralOp]> {
  let summary = "'always @' block";
  let description = "See SV Spec 9.2, and 9.4.2.2.";

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins TypedArrayAttrBase<EventControlAttr, "events">:$events,
                       Variadic<I1>:$clocks);
  let results = (outs);

  let assemblyFormat = "custom<EventList>($events, $clocks) $body attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ArrayRef<EventControl>":$event, "ArrayRef<Value>":$cond,
                      CArg<"std::function<void()>", "{}">:$bodyCtor)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }

    struct Condition {
      EventControl event;
      Value value;
    };

    size_t getNumConditions() { return getEvents().size(); }
    Condition getCondition(size_t idx);
  }];

  // Check that we have the same number of events and conditions.
  let hasVerifier = 1;
}

def AlwaysCombOp : SVOp<"alwayscomb", [SingleBlock, NoTerminator,
                                       NoRegionArguments,
                                       RecursiveMemoryEffects,
                                       RecursivelySpeculatable,
                                       ProceduralRegion, NonProceduralOp]> {
  let summary = "'alwayscomb block";
  let description = "See SV Spec 9.2, and 9.2.2.2.";

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "$body attr-dict";

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"std::function<void()>", "{}">:$bodyCtor)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def NoReset: I32EnumAttrCase<"NoReset", 0, "noreset">;
def SyncReset: I32EnumAttrCase<"SyncReset", 1, "syncreset">;
def AsyncReset: I32EnumAttrCase<"AsyncReset", 2, "asyncreset">;

def ResetTypeAttr : I32EnumAttr<"ResetType", "reset type",
                                   [NoReset, SyncReset, AsyncReset]>;


def AlwaysFFOp : SVOp<"alwaysff", [SingleBlock, NoTerminator, NoRegionArguments,
                                   RecursiveMemoryEffects, RecursivelySpeculatable,
                                   ProceduralRegion, NonProceduralOp]> {
  let summary = "'alwaysff @' block with optional reset";
  let description = [{
    alwaysff blocks represent always_ff verilog nodes, which enforce inference
    of registers.  This block takes a clock signal and edge sensitivity and
    reset type.  If the reset type is anything but 'noreset', the block takes a
    reset signal, reset sensitivity, and reset block.  Appropriate if conditions
    are generated in the output code based on the reset type.  A negative-edge,
    asynchronous reset will check the inverse of the reset condition
    (if (!reset) begin resetblock end) to match the sensitivity.
  }];

  let regions = (region SizedRegion<1>:$bodyBlk, AnyRegion:$resetBlk);
  let arguments = (ins EventControlAttr:$clockEdge, I1:$clock,
                       DefaultValuedAttr<ResetTypeAttr,
                                         "ResetType::NoReset">:$resetStyle,
                       OptionalAttr<EventControlAttr>:$resetEdge,
                       Optional<I1>:$reset);
  let results = (outs);

  let assemblyFormat = [{
    `(` $clockEdge $clock `)` $bodyBlk
    ( `(` $resetStyle `:` $resetEdge^ $reset `)` $resetBlk )? attr-dict
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "EventControl":$clockEdge, "Value":$clock,
                      CArg<"std::function<void()>", "{}">:$bodyCtor)>,
    OpBuilder<(ins "EventControl":$clockEdge, "Value":$clock,
                      "ResetType":$resetStyle,
                      "EventControl":$resetEdge, "Value":$reset,
                      CArg<"std::function<void()>", "{}">:$bodyCtor,
                      CArg<"std::function<void()>", "{}">:$resetCtor)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBodyBlk().front(); }
    Block *getResetBlock() { return &getResetBlk().front(); }
  }];
}


def InitialOp : SVOp<"initial", [SingleBlock, NoTerminator, NoRegionArguments,
                                 RecursiveMemoryEffects,
                                 RecursivelySpeculatable,
                                 ProceduralRegion, NonProceduralOp]> {
  let summary = "'initial' block";
  let description = "See SV Spec 9.2.1.";

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "$body attr-dict";

  // TODO: ODS forces using a custom builder just to get the region terminator
  // implicitly installed.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins CArg<"std::function<void()>", "{}">:$bodyCtor)>
  ];

  let extraClassDeclaration = [{
    Block *getBodyBlock() { return &getBody().front(); }
  }];
}

def CaseStmt: I32EnumAttrCase<"CaseStmt", 0, "case">;
def CaseXStmt: I32EnumAttrCase<"CaseXStmt", 1, "casex">;
def CaseZStmt: I32EnumAttrCase<"CaseZStmt", 2, "casez">;

def CaseStmtTypeAttr : I32EnumAttr<"CaseStmtType", "case type",
                                   [CaseStmt, CaseXStmt, CaseZStmt]>;

def ValidationQualifierPlain: I32EnumAttrCase<"ValidationQualifierPlain", 0, "plain">;
def ValidationQualifierUnique: I32EnumAttrCase<"ValidationQualifierUnique", 1, "unique">;
def ValidationQualifierUnique0: I32EnumAttrCase<"ValidationQualifierUnique0", 2, "unique0">;
def ValidationQualifierPriority: I32EnumAttrCase<"ValidationQualifierPriority", 3, "priority">;

def ValidationQualifierTypeEnum: I32EnumAttr<"ValidationQualifierTypeEnum", "validation qualifier type",
              [ValidationQualifierPlain, ValidationQualifierUnique,
               ValidationQualifierUnique0, ValidationQualifierPriority]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::circt::sv";
}

def ValidationQualifierTypeAttr: EnumAttr<SVDialect, ValidationQualifierTypeEnum, "validation_qualifier">;

def CaseOp : SVOp<"case", [SingleBlock, NoTerminator, NoRegionArguments,
                           ProceduralRegion, ProceduralOp]> {
  let summary = "'case (cond)' block";
  let description = "See SystemVerilog 2017 12.5.";

  let regions = (region VariadicRegion<SizedRegion<1>>:$caseRegions);
  let arguments = (ins DefaultValuedAttr<CaseStmtTypeAttr,
                                         "CaseStmtType::CaseStmt">:$caseStyle,
                       AnyType:$cond, ArrayAttr:$casePatterns,
                       DefaultValuedAttr<ValidationQualifierTypeAttr,
                       "ValidationQualifierTypeEnum::ValidationQualifierPlain">:
                      $validationQualifier);
  let results = (outs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;

  let builders = [
    /// This ctor allows you to build a CaseZ with some number of cases, getting
    /// a callback for each case.
    OpBuilder<(ins "CaseStmtType":$caseStyle,
                   "ValidationQualifierTypeEnum":$validationQualifier,
                   "Value":$cond, "size_t":$numCases,
                   "std::function<std::unique_ptr<CasePattern>(size_t)>":$caseCtor)>,
    OpBuilder<(ins "CaseStmtType":$caseStyle, "Value":$cond, "size_t":$numCases,
               "std::function<std::unique_ptr<CasePattern>(size_t)>":$caseCtor), [{
      build($_builder, $_state, caseStyle,
            ValidationQualifierTypeEnum::ValidationQualifierPlain, cond,  numCases,
            caseCtor);
    }]>
  ];

  let extraClassDeclaration = [{
    SmallVector<CaseInfo, 4> getCases();

    // void addCase(...)
  }];
}

//===----------------------------------------------------------------------===//
// Assignment Related Statements
//===----------------------------------------------------------------------===//

def AssignOp : SVOp<"assign", [InOutTypeConstraint<"src", "dest">,
                                 NonProceduralOp]> {
  let summary = "Continuous assignment";
  let description = [{
    A SystemVerilog assignment statement 'x = y;'.
    These occur in module scope.  See SV Spec 10.3.2.
  }];
  let arguments = (ins InOutType:$dest, InOutElementType:$src);
  let results = (outs);
  let assemblyFormat = "$dest `,` $src attr-dict `:` qualified(type($src))";
}

def BPAssignOp : SVOp<"bpassign", [InOutTypeConstraint<"src", "dest">,
                                   ProceduralOp]> {
  let summary = "Blocking procedural assignment";
  let description = [{
    A SystemVerilog blocking procedural assignment statement 'x = y;'.  These
    occur in initial, always, task, and function blocks.  The statement is
    executed before any following statements are. See SV Spec 10.4.1.
  }];
  let hasCanonicalizeMethod = true;
  let arguments = (ins InOutType:$dest, InOutElementType:$src);
  let results = (outs);
  let assemblyFormat = "$dest `,` $src  attr-dict `:` qualified(type($src))";
  let hasVerifier = 1;
}

def PAssignOp : SVOp<"passign", [InOutTypeConstraint<"src", "dest">,
                                 ProceduralOp]> {
  let summary = "Nonblocking procedural assignment";
  let description = [{
    A SystemVerilog nonblocking procedural assignment statement 'x <= y;'.
    These occur in initial, always, task, and function blocks.  The statement
    can be scheduled without blocking procedural flow.  See SV Spec 10.4.2.
  }];
  let hasCanonicalizeMethod = true;
  let arguments = (ins InOutType:$dest, InOutElementType:$src);
  let results = (outs);
  let assemblyFormat = "$dest `,` $src  attr-dict `:` qualified(type($src))";
  let hasVerifier = 1;
}



//===----------------------------------------------------------------------===//
// Other Statements
//===----------------------------------------------------------------------===//

def ForceOp : SVOp<"force", [InOutTypeConstraint<"src", "dest">,
                                   ProceduralOp]> {
  let summary = "Force procedural statement";
  let description = [{
    A SystemVerilog force procedural statement 'force x = y;'.  These
    occur in initial, always, task, and function blocks.
    A force statement shall override a procedural assignment until
    a release statement is executed on the variable.
    The left-hand side of the assignment can be a variable, a net,
    a constant bit-select of a vector net, a part-select of a vector
    net or a concatenation. It cannot be a memory word or a bit-select
    or part-select of a vector variable. See SV Spec 10.6.2.
  }];
  let arguments = (ins InOutType:$dest, InOutElementType:$src);
  let results = (outs);
  let assemblyFormat = "$dest `,` $src  attr-dict `:` qualified(type($src))";
}

def ReleaseOp : SVOp<"release", [ProceduralOp]> {
  let summary = "Release procedural statement";
  let description = [{
    Release is used in conjunction with force. When released,
    then if the variable does not currently have an active assign
    procedural continuous assignment, the variable shall not immediately
    change value. The variable shall maintain its current value until
    the next procedural assignment or procedural continuous assignment
    to the variable. Releasing a variable that currently has an
    active assign procedural continuous assignment shall immediately
    reestablish that assignment. See SV Spec 10.6.2.
  }];
  let arguments = (ins InOutType:$dest);
  let results = (outs);
  let assemblyFormat = "$dest attr-dict `:` qualified(type($dest))";
}

def AliasOp : SVOp<"alias"> {
  let summary = "SystemVerilog 'alias' statement";
  let description = [{
    An alias statement declares multiple names for the same physical net, or
    bits within a net.  Aliases always have at least two operands.
  }];

  let arguments = (ins Variadic<InOutType>:$aliases);
  let results = (outs);

  let assemblyFormat = "$aliases attr-dict `:` qualified(type($aliases))";

  // Verify that we have at least two operands.
  let hasVerifier = 1;
}

def FWriteOp : SVOp<"fwrite", [ProceduralOp]> {
  let summary = "'$fwrite' statement";

  let arguments = (ins I32:$fd,
                       StrAttr:$format_string,
                       Variadic<AnyType>:$substitutions);
  let results = (outs);

  let assemblyFormat = [{
    $fd `,` $format_string attr-dict (`(` $substitutions^ `)` `:`
    qualified(type($substitutions)))?
  }];
}

def VerbatimOp : SVOp<"verbatim"> {
  let summary = "Verbatim opaque text emitted inline.";
  let description = [{
    This operation produces opaque text inline in the SystemVerilog output.

    sv.verbatim allows operand substitutions with {{0}} syntax.
  }];
  let arguments = (ins StrAttr:$format_string, Variadic<AnyType>:$substitutions,
                  DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols);

  let assemblyFormat = [{
    $format_string (`(` $substitutions^ `)` `:`
    qualified(type($substitutions)))? attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Twine":$formatString),
                 "build(odsBuilder, odsState, "
                 "odsBuilder.getStringAttr(formatString), ValueRange{},"
                 "odsBuilder.getArrayAttr({}));">,
    OpBuilder<(ins "StringAttr":$formatString),
                 "build(odsBuilder, odsState, formatString, ValueRange{},"
                 "odsBuilder.getArrayAttr({}));">
  ];
}

//===----------------------------------------------------------------------===//
// Bind Statements
//===----------------------------------------------------------------------===//

def BindOp : SVOp<"bind",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "indirect instantiation statement";
  let description = [{
    Indirectly instantiate a module from the context of another module. BindOp
    pairs with a `hw.instance` (identified by a `boundInstance` symbol) which
    tracks all information except the emission point for the bind.  BindOp also
    tracks the `instanceModule` symbol for the `hw.module` that contains the
    `hw.instance` to accelerate symbol lookup.

    See 23.11 of SV 2017 spec for more information about bind.
  }];

  let arguments = (ins InnerRefAttr:$instance);
  let results = (outs);

  let assemblyFormat = "$instance attr-dict";
  let builders = [
    OpBuilder<(ins "StringAttr":$mod, "StringAttr":$name)>
  ];
  let extraClassDeclaration = [{
    /// Lookup the instance for the bind.  This returns null on invalid IR.
    hw::InstanceOp getReferencedInstance(const hw::HWSymbolCache *cache = nullptr);
  }];
}

def BindInterfaceOp : SVOp<"bind.interface",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "indirectly instantiate an interface";
  let description = [{
    Indirectly instantiate an interface in the context of another module. This
    operation must pair with a `sv.interface.instance`.
  }];
  let arguments = (ins InnerRefAttr:$instance);
  let results = (outs);

  let assemblyFormat = "$instance attr-dict";
  let extraClassDeclaration = [{
    sv::InterfaceInstanceOp
    getReferencedInstance(const hw::HWSymbolCache *cache = nullptr);
  }];
}

//===----------------------------------------------------------------------===//
// Simulation Control Tasks
//===----------------------------------------------------------------------===//

/// A number passed to `$stop` and `$finish` that indicates the level of
/// verbosity of the diagnostic message to be printed (see section 20.2 of IEEE
/// 1800-2017):
///
/// - `0`: Prints nothing
/// - `1`: Prints simulation time and location (default)
/// - `2`: Prints simulation time, location, and statistics about the memory and
///   CPU time used in simulation
def VerbosityIntAttr : DefaultValuedAttr<ConfinedAttr<I8Attr,
  [IntMinValue<0>, IntMaxValue<2>]>, "1">;

/// Commonalities between `StopOp` and `FinishOp`.
class FinishOrStopOp<string mnemonic, list<Trait> traits = []> :
    SVOp<mnemonic, [ProceduralOp] # traits> {
  let arguments = (ins VerbosityIntAttr:$verbosity);
  let assemblyFormat = "$verbosity attr-dict";

  // First line intentionally left blank.
  string verbosityDescription = [{
    The optional `verbosity` parameter controls how much diagnostic information
    is printed when the system task is executed (see section 20.2 of IEEE
    1800-2017):

    - `0`: Prints nothing
    - `1`: Prints simulation time and location (default)
    - `2`: Prints simulation time, location, and statistics about the memory and
      CPU time used in simulation
  }];
}

def StopOp : FinishOrStopOp<"stop"> {
  let summary = "`$stop` system task";
  let description = [{
    Causes the simulation to be suspended. Does not terminate the simulator.
  }] # verbosityDescription;
}

def FinishOp : FinishOrStopOp<"finish"> {
  let summary = "`$finish` system task";
  let description = [{
    Stops the simulation and exits/terminates the simulator process. In practice
    most GUI-based simulators will show a prompt to the user offering them an
    opportunity to not close the simulator altogether.

    Other tasks such as `$exit` or `$fatal` implicitly call this system task.
  }] # verbosityDescription;
}

def ExitOp : SVOp<"exit", [ProceduralOp]> {
  let summary = "`$exit` system task";
  let description = [{
    Waits for all `program` blocks to complete and then makes an implicit call
    to `$finish` with default verbosity (level 1) to conclude the simulation.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Severity Message Tasks
//===----------------------------------------------------------------------===//

def FatalOp : SVOp<"fatal", [ProceduralOp]> {
  let summary = "`$fatal` severity message task";
  let description = [{
    Generates a run-time fatal error which terminates the simulation with an
    error code. Makes an implicit call to `$finish`, forwarding the `verbosity`
    operand. If present, the optional message is printed with any additional
    operands interpolated into the message string.
  }];
  let arguments = (ins
    VerbosityIntAttr:$verbosity,
    OptionalAttr<StrAttr>:$message, Variadic<AnyType>:$substitutions);
  let assemblyFormat = [{
    $verbosity
    (`,` $message^ (`(` $substitutions^ `)` `:`
    qualified(type($substitutions)))?)? attr-dict
  }];
  let builders = [
    OpBuilder<(ins CArg<VerbosityIntAttr.returnType,"1">:$verbosity),
      "build(odsBuilder, odsState, verbosity, StringAttr{}, ValueRange{});">,
    OpBuilder<(ins VerbosityIntAttr.returnType:$verbosity,
      "mlir::StringAttr":$message),
      "build(odsBuilder, odsState, verbosity, message, ValueRange{});">,
  ];
}

/// Commonalities between `ErrorOp`, `WarningOp`, and `InfoOp`.
class NonfatalMessageOp<string mnemonic, list<Trait> traits = []> :
    SVOp<mnemonic, [ProceduralOp] # traits> {
  string messageDescription = [{
    If present, the optional message is printed with any additional operands
    interpolated into the message string.
  }];
  let arguments = (ins
    OptionalAttr<StrAttr>:$message, Variadic<AnyType>:$substitutions);
  let assemblyFormat = [{
    ($message^ (`(` $substitutions^ `)` `:` qualified(type($substitutions)))?)?
    attr-dict
  }];
  let builders = [
    OpBuilder<(ins),
      "build(odsBuilder, odsState, StringAttr{}, ValueRange{});">,
    OpBuilder<(ins "mlir::StringAttr":$message),
      "build(odsBuilder, odsState, message, ValueRange{});">,
  ];
}

def ErrorOp : NonfatalMessageOp<"error"> {
  let summary = "`$error` severity message task";
  let description = [{
    This system task indicates a run-time error.
  }] # messageDescription;
}

def WarningOp : NonfatalMessageOp<"warning"> {
  let summary = "`$warning` severity message task";
  let description = [{
    This system task indicates a run-time warning.
  }] # messageDescription;
}

def InfoOp : NonfatalMessageOp<"info"> {
  let summary = "`$info` severity message task";
  let description = [{
    This system task indicates a message with no specific severity.
  }] # messageDescription;
}

def DepositOp : SVOp<"nonstandard.deposit",
    [ProceduralOp, VendorExtension, InOutTypeConstraint<"src", "dest">]> {
  let summary = "`$deposit` system task";
  let description = [{
    This system task sets the value of a net or variable, but doesn't hold it.
    This is a common simulation vendor extension.
  }];
  let arguments = (ins InOutType:$dest, InOutElementType:$src);
  let results = (outs);
  let assemblyFormat = "$dest `,` $src  attr-dict `:` qualified(type($src))";
}


//===----------------------------------------------------------------------===//
// Memory loading system tasks
//===----------------------------------------------------------------------===//

def MemBaseBin: I32EnumAttrCase<"MemBaseBin", 0>;
def MemBaseHex: I32EnumAttrCase<"MemBaseHex", 1>;
def MemBaseTypeAttr :
    I32EnumAttr<"MemBaseTypeAttr", "the numeric base of a memory file",
                [MemBaseBin, MemBaseHex]>;

def ReadMemOp : SVOp<"readmem", [ProceduralOp]> {
  let summary = "load a memory from a file in either binary or hex format";
  let description = [{
    Load a memory from a file using either `$readmemh` or `$readmemb` based on
    an attribute.

    See Section 21.4 of IEEE 1800-2017 for more information.
  }];
  let arguments =
    (ins InOutType:$dest, StrAttr:$filename, MemBaseTypeAttr:$base);
  let results = (outs);
  let assemblyFormat =
    "$dest `,` $filename `,` $base attr-dict `:` qualified(type($dest))";
}

def ForOp : SVOp<"for", [SingleBlock, NoTerminator, ProceduralOp,
                         ProceduralRegion, OpAsmOpInterface,
                         AllTypesMatch<["lowerBound", "upperBound", "step"]>]> {
  let summary = "System verilog for loop";

  let description = [{
    The `sv.for` operation in System Verilog defines a for statement that requires
    three SSA operands: `lowerBounds`, `upperBound`, and `step`. It functions
    similarly to `scf.for`, where the loop iterates the induction variable from
    `lowerBound` to `upperBound` with a step size of `step`, i.e:

    ```
     for (logic ... indVar = lowerBound; indVar < upperBound; indVar += step) begin
     end
    ```

    It's important to note that since we are using a bit precise type instead of a Verilog
    `integer` type, users must be cautious about potential overflow. For example, if
    you wish to iterate over all 2-bit values, you must use a 3-bit value as the
    induction variable type.
  }];

  let regions = (region SizedRegion<1>:$body);
  let arguments = (ins HWIntegerType:$lowerBound,
                       HWIntegerType:$upperBound,
                       HWIntegerType:$step,
                       DefaultValuedAttr<StrAttr, "{}">:$inductionVarName);
  let results = (outs);
  let builders = [
      OpBuilder<(ins "Value":$lowerBound, "Value":$upperBound,
                     "Value":$step, "StringRef":$inductionVarName,
                     "llvm::function_ref<void(BlockArgument)>": $thenCtor)>,
      OpBuilder<(ins "int64_t":$lowerBound, "int64_t":$upperBound,
                     "int64_t":$step, "IntegerType": $type,
                     "StringRef":$inductionVarName,
                     "llvm::function_ref<void(BlockArgument)>": $thenCtor)>
  ];
  let hasCustomAssemblyFormat = 1;
  let hasCanonicalizeMethod = 1;
  let extraClassDeclaration = [{
    Block* getBodyBlock() { return &getBody().front(); }
    Value getInductionVar() { return getBodyBlock()->getArgument(0); }
    void getAsmBlockArgumentNames(mlir::Region &region,
                                  mlir::OpAsmSetValueNameFn setNameFn);
  }];
}

//===----------------------------------------------------------------------===//
// Macros
//===----------------------------------------------------------------------===//


def MacroDeclOp : SVOp<"macro.decl", [Symbol]> {
  let summary = "System verilog macro declaration";

  let description = [{
    The `sv.macro.def` declares a macro in System Verilog.  This is a 
    declaration; the body of the macro, which produces a verilog macro 
    definition is created with a `macro.def` operation.
   
    Lacking args will be a macro without "()".  An empty args will be an empty "()".

    The verilog name is the spelling of the macro when emitting verilog.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name, 
                       OptionalAttr<StrArrayAttr>:$args,
                       OptionalAttr<StrAttr>:$verilogName);
  let results = (outs);

  let assemblyFormat = [{
    $sym_name (`[` $verilogName^ `]`)? (`(` $args^ `)`)?  attr-dict
  }];
}


def MacroDefOp : SVOp<"macro.def", 
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "System verilog macro definition";

  let description = [{
    The `sv.macro.def` defines a macro in System Verilog which optionally takes
    a body.

    This is modeled similarly to verbatim in that the contents of the macro are
    opaque (plain string).  Given the general power of macros, this op does not
    try to capture a return type.
  
    This operation produces a definition for the macro declaration referenced by 
    `sym_name`.  Argument lists are picked up from that operation.

    sv.macro.def allows operand substitutions with {{0}} syntax.
  }];

  let arguments = (ins FlatSymbolRefAttr:$macroName, 
                       StrAttr:$format_string,
                       DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols);
  let results = (outs);

  let builders = [
    OpBuilder<(ins "StringRef":$macroName, "StringRef":$format_string),
                "build($_builder, $_state, macroName, format_string,"
                "odsBuilder.getArrayAttr({}));">
  ];

  let assemblyFormat = [{
    $macroName $format_string (`(` $symbols^ `)`)? attr-dict
  }];

  let extraClassDeclaration = [{
    MacroDeclOp getReferencedMacro(const hw::HWSymbolCache *cache);
  }];
}
