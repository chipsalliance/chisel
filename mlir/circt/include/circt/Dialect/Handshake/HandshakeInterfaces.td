//===- HandshakeInterfaces.td - Handshake interfaces -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for the structured interface for Handshake ops.
//
//===----------------------------------------------------------------------===//

#ifndef HANDSHAKE_OP_INTERFACES
#define HANDSHAKE_OP_INTERFACES

include "mlir/IR/OpBase.td"

def SOSTInterface : OpInterface<"SOSTInterface"> {
  let description = [{
      Sized Operation with Single Type (SOST).
      
      These are operations whose operands all have the same type and which have
      an integer size property, be it the number of operation operands (e.g.,
      for a merge) or the number of operation results (e.g., for a fork).
  }];

  let methods = [
    InterfaceMethod<[{
        Get the data type associated to the operation.
        The default implementation of this method simply returns the type of
        the first operation operand.
      }],
      "Type", "getDataType", (ins), "",
      [{
        auto concreteOp = cast<ConcreteOp>($_op);
        return concreteOp->getOperands().front().getType();
      }]
    >,
    InterfaceMethod<[{
        Get the size associated to the operation.
        The default implementation of this method simply returns the number of
        operation operands.
      }],
      "unsigned", "getSize", (ins), "",
      [{
        auto concreteOp = cast<ConcreteOp>($_op);
        return concreteOp->getNumOperands();
      }]
    >,
    InterfaceMethod<[{
        Determine whether the operation is a control operation.
        The default implementation of this method assumes that the operation
        is a control operation if and only if its associated data type is a
        NoneType.
      }],
      "bool", "sostIsControl", (ins), "",
      [{
        auto concreteOp = cast<ConcreteOp>($_op);
        // The operation is a control operation if its single data type is a 
        // NoneType.
        return concreteOp.getDataType().template isa<NoneType>();
      }]
    >,
    InterfaceMethod<[{
        Print the "SOST characteristics" of an operation.
        If the `explicitSize` parameter is set to true, then the method prints
        the operation's size (in the SOST sense) between square brackets before
        printing the operation's operands, attributes, and data type.
      }],
      "void", "sostPrint", (ins 
        "mlir::OpAsmPrinter &": $printer, "bool": $explicitSize
      ), "", 
      [{
        auto concreteOp = cast<ConcreteOp>($_op);

        if (explicitSize) {
          printer << " [" << concreteOp.getSize() << "]";
        }
        printer << " " << concreteOp->getOperands();
        printer.printOptionalAttrDict(concreteOp->getAttrs());
        printer << " : " << concreteOp.getDataType();
      }]
    >
  ];

  let verify = [{
    auto concreteOp = cast<ConcreteOp>($_op);

    // SOST operation's size must be at least one
    if (concreteOp.getSize() < 1) {
      return concreteOp.emitOpError(
        "SOST operation's size must be at least 1, but has size ") 
        << concreteOp.getSize();
    }

    // SOST operation's operands must all have the same type
    auto dataType = concreteOp.getDataType();
    for (auto operand : concreteOp->getOperands())
      if (operand.getType() != dataType) 
        return concreteOp.emitOpError("SOST operation reports having data type ")
          << dataType << ", but one operand has type " << operand.getType();
    
    return success();
  }];
}

def MergeLikeOpInterface : OpInterface<"MergeLikeOpInterface"> {
  let description = [{
     Some handshake operations can have predecessors in other
     blocks. This is primarily useful for verification purposes during
     lowering from other dialect, such as the standard CDFG dialect.
  }];

  let methods = [
    InterfaceMethod<[{
        Returns an operand range over the data signals being merged.
      }],
      "OperandRange", "getDataOperands", (ins)
    >,
  ];

  let verify = [{
    auto concreteOp = cast<ConcreteOp>($_op);

    auto operands = concreteOp.getDataOperands();

    if (!operands.size())
      return concreteOp.emitOpError("must have at least one data operand");

    Type resultType = $_op->getResult(0).getType();

    for (auto operand : operands)
      if (operand.getType() != resultType)
        return concreteOp.emitOpError("operand has type ") << operand.getType()
            << ", but result has type " << resultType;

    return success();
  }];
}

def GeneralOpInterface : OpInterface<"GeneralOpInterface"> {
  let description =
      [{"Simulate the Execution of ops. The op takes a set of input values and "
        "returns the corresponding outputs assuming the precondition to "
        "execute holds."}];

  let methods = [
    InterfaceMethod<
        "Simulate the Execution of the general op with given inputs", "void",
        "execute",
        (ins "std::vector<llvm::Any> &" : $ins,
           "std::vector<llvm::Any> &" : $outs)>,
  ];
}

def ExecutableOpInterface : OpInterface<"ExecutableOpInterface"> {
  let description = [{"Simulate the Execution of ops"}];

  let methods = [
    InterfaceMethod<
        "The op checks its precondition to execute. If the precondition holds, "
        "the op executes with the given input values and returns zero. "
        "Otherwise, it returns a false value to suggest the simulator to "
        "reschedule. The execution interface of hpx ops take the following "
        "arguments: "
        "bool = returns whether the op has been executed;"
        "valueMap = a map of all values in the code;"
        "memoryMap = a map of memory ops to simualte;"
        "timeMap = a map of the last arrival time of all values;"
        "store = The store associates each allocation in the program"
        "(represented by a int) with a vector of values which can be"
        "accessed by it."
        "scheduleList = a list of values to be scheduled.",
        "bool", "tryExecute",
        (ins "llvm::DenseMap<mlir::Value, llvm::Any> &" : $valueMap,
           "llvm::DenseMap<unsigned, unsigned> &" : $memoryMap,
           "llvm::DenseMap<mlir::Value, double> &" : $timeMap,
           "std::vector<std::vector<llvm::Any>> &" : $store,
           "std::vector<mlir::Value> &" : $scheduleList)>,
  ];
}

def MemoryOpInterface : OpInterface<"MemoryOpInterface"> {
  let description =
      [{"Allocate the memory to the meory map in the simulation. "}];

  let methods = [
    InterfaceMethod<
        "Simulate the memory allocation in the memoryMap", "bool",
        "allocateMemory",
        (ins "llvm::DenseMap<unsigned, unsigned> &" : $memoryMap,
    "std::vector<std::vector<llvm::Any>> &" : $store,
    "std::vector<double> &" : $storeTimes)>,
  ];
}

def NamedIOInterface : OpInterface<"NamedIOInterface"> {
  let description =
      [{"Provides detailed names for the operands and results of an operation."}];

  let methods = [
    InterfaceMethod<
        "Get the name of an operand.",
        "std::string", "getOperandName", (ins "unsigned" : $idx),
        "",
        [{
          // Return the default name of an operand.
          return "in" + std::to_string(idx);
        }]>,
    InterfaceMethod<
        "Get the name of a result.",
        "std::string", "getResultName", (ins "unsigned" : $idx),
        "",
        [{
          // Return the default name of a result.
          return "out" + std::to_string(idx);
        }]>
  ];
}

def ControlInterface : OpInterface<"ControlInterface"> {
  let description =
      [{"Provides information on whether this operation is a control operation."}];

  let methods = [
    InterfaceMethod<
        "Returns true if this operation is a control operation.",
        "bool",
        "isControl",
        (ins),
        "",
        [{
          // Implemented outside of interface due to dependency on
          // SOSTInterface being declared at the time where this method is
          // defined.
          return isControlOpImpl($_op);
        }]>];
}

def HasClock : NativeOpTrait<"HasClock">;

#endif // HANDSHAKE_OP_INTERFACES
