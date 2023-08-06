//===- CalyxInterfaces.td - Calyx Interfaces -------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the Calyx dialect.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

// Op interface for groups.
def GroupOpInterface : OpInterface<"GroupInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for Calyx Groups.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the symbol name of the group.",
      "mlir::StringAttr",
      "symName",
      (ins), [{
        Operation* op = $_op;
        return op->getAttrOfType<mlir::StringAttr>("sym_name");
      }]>,
    InterfaceMethod<
      "This returns the body of the group.",
      "Block*",
      "getBody",
      (ins), [{
        Region* region = &$_op->getRegion(0);
        assert(region->hasOneBlock() && "The body should have one Block.");
        return &region->front();
      }]>,
    InterfaceMethod<
      "Determines whether the given port is driven by this group.",
      "LogicalResult",
      "drivesPort",
      (ins "Value":$port)>,
    InterfaceMethod<
      "Determines whether all the given ports are driven by this group.",
      "LogicalResult",
      "drivesAllPorts",
      (ins "ValueRange":$ports)>,
    InterfaceMethod<
      "Determines whether any of the given ports are driven by this group.",
      "LogicalResult",
      "drivesAnyPort",
      (ins "ValueRange":$ports)>,
    InterfaceMethod<
      "Determines whether any of the given ports are read by this group.",
      "LogicalResult",
      "readsAnyPort",
      (ins "ValueRange":$ports)>
  ];
  let verify = [{ return verifyGroupInterface(op); }];
}

/// Op Interface for cells.
def CellOpInterface : OpInterface<"CellInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for Calyx Cells. Cells consist
    of primitives and instances of components.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the port names associated with the cell.",
      "SmallVector<StringRef>", "portNames"
    >,
    InterfaceMethod<
      "This returns the port directions associated with the cell.",
      "SmallVector<circt::calyx::Direction>", "portDirections"
    >,
    InterfaceMethod<
      "This returns the port attributes associated with the cell.",
      "SmallVector<DictionaryAttr>", "portAttributes"
    >,
    InterfaceMethod<
      "This returns the instance name of the cell.",
      "StringRef",
      "instanceName",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getOperation()->template getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName()).getValue();
      }]
    >,
    InterfaceMethod<
      "This returns the index of the port within the result values of this cell.",
      "unsigned",
      "portIndex",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        for (auto it : llvm::enumerate($_op->getResults()))
          if (it.value() == port)
            return it.index();
        assert(false && "argument port not found in this cell.");
        return  {};
      }]
    >,
    InterfaceMethod<
      "This returns the PortInfo associated with all of the ports of a cell.",
      "SmallVector<circt::calyx::PortInfo>",
      "getPortInfo",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<circt::calyx::PortInfo> ports;
        MLIRContext* context = $_op->getContext();
        auto zippedPortInfo = llvm::zip(
          $_op->getResults(),
          $_op.portDirections(),
          $_op.portNames(),
          $_op.portAttributes()
        );
        for (auto&& [result, direction, name, attributes] : zippedPortInfo)
          ports.push_back(PortInfo{
              StringAttr::get(context, name),
              result.getType(),
              direction,
              attributes
          });
        return ports;
    }]
    >,
    InterfaceMethod<
      "This returns the PortInfo associated with the port of a cell.",
      "circt::calyx::PortInfo",
      "portInfo",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return getPortInfo()[portIndex(port)]; }]
    >,
    InterfaceMethod<
      "This returns the direction of a given port of a cell.",
      "circt::calyx::Direction",
      "direction",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return portInfo(port).direction; }]
    >,
    InterfaceMethod<
      "This returns the name of a given port of a cell.",
      "StringRef",
      "portName",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return portInfo(port).name.getValue(); }]
    >,
    InterfaceMethod<
      "This returns the attributes of a given port of a cell.",
      "DictionaryAttr",
      "portDictionaryAttr",
      (ins "Value":$port),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{ return portInfo(port).attributes; }]
    >,
    InterfaceMethod<
      "This returns the input ports of a cell.",
      "SmallVector<Value>",
      "getInputPorts",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<Value> ports;
        for (auto&& [result, direction] : llvm::zip($_op->getResults(), $_op.portDirections()))
          if (direction == calyx::Direction::Input)
            ports.push_back(result);
        return ports;
      }]
    >,
    InterfaceMethod<
      "This returns the output ports of a cell.",
      "SmallVector<Value>",
      "getOutputPorts",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        SmallVector<Value> ports;
        for (auto&& [result, direction] : llvm::zip($_op->getResults(), $_op.portDirections()))
          if (direction == calyx::Direction::Output)
            ports.push_back(result);
        return ports;
      }]
    >,
    InterfaceMethod<
      "This returns the ports of a cell, filtering out any interface ports.",
      "SmallVector<Value>",
      "filterInterfacePorts",
      (ins "calyx::Direction":$dir),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        auto isInterfacePort = [](const PortInfo& port) {
          return port.hasAttribute("go")   || port.hasAttribute("reset") ||
                 port.hasAttribute("done") || port.hasAttribute("clk");
        };

        SmallVector<Value> filteredPorts;
        for (auto&& [result, port] : llvm::zip($_op->getResults(), $_op.getPortInfo())) {
          if (port.direction != dir || !isInterfacePort(port))
            continue;
          filteredPorts.push_back(result);
        }
        return filteredPorts;
      }]
    >,
    InterfaceMethod<
      "This returns true iff the cell is combinational",
      "bool", "isCombinational"
    >
  ];

  let verify = [{ return verifyCell(op); }];
}

def ComponentOpInterface : OpInterface<"ComponentInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for Calyx Components.
  }];

  let methods = [
    InterfaceMethod<
      "This returns the port names associated with the component.",
      "ArrayAttr", "getPortNames"
    >,
    InterfaceMethod<
      "This returns the port info associated with the component.",
      "SmallVector<circt::calyx::PortInfo>", "getPortInfo"
    >,
    InterfaceMethod<
      "This returns the input port info associated with the component.",
      "SmallVector<circt::calyx::PortInfo>", "getInputPortInfo"
    >,
    InterfaceMethod<
      "This returns the output port info associated with the component.",
      "SmallVector<circt::calyx::PortInfo>", "getOutputPortInfo"
    >,
    InterfaceMethod<
      "This returns the name of the component.",
      "StringRef", "getName"
    >,
    InterfaceMethod<
      "This returns true iff the component is combinational.",
      "bool", "isComb"
    >,
    InterfaceMethod<
      "This returns the body region of the component.",
      "Region*", "getRegion"
    >,
    InterfaceMethod<
      "This returns the body block of the component.",
      "Block*", "getBodyBlock"
    >
  ];

  let verify = [{ return verifyComponent(op); }];
}

def IfOpInterface : OpInterface<"IfInterface"> {
  let cppNamespace = "::circt::calyx";

  let description = [{
    This is an op interface for Calyx If control ops. Implemented
    by Dynamic and Static If ops.
  }];

  let methods = [
    InterfaceMethod<
      "This returns true if the then body exists.",
      "bool",
      "thenBodyExists",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Region* region = &$_op->getRegion(0);
        if (region == nullptr)
          return false;
        return !region->empty();
      }]
    >,
    InterfaceMethod<
      "This returns true if the else body exists.",
      "bool",
      "elseBodyExists",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Region* region = &$_op->getRegion(1);
        if (region == nullptr)
          return false;
        return !region->empty();
      }]
    >,
    InterfaceMethod<
      "This returns the then body block.",
      "Block*",
      "getThenBody",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(thenBodyExists() && "Then region does not exist.");
        Region* region = &$_op->getRegion(0);
        assert(region->hasOneBlock() && "The Then body should have one Block.");
        return &region->front();
      }]
    >,
    InterfaceMethod<
      "This returns the else body block.",
      "Block*",
      "getElseBody",
      (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(elseBodyExists() && "Else region does not exist.");
        Region* region = &$_op->getRegion(1);
        assert(region->hasOneBlock() && "The Else body should have one Block.");
        return &region->front();
      }]
    >
  ];

  let verify = [{ return verifyIf(op); }];
}
