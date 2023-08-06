//===- ESIInterfaces.td - ESI Interfaces -----------------*- tablegen -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the interfaces in the ESI dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_INTERFACES_TD
#define CIRCT_DIALECT_ESI_INTERFACES_TD

include "mlir/IR/OpBase.td"

def ChannelOpInterface : OpInterface<"ChannelOpInterface"> {
  let cppNamespace = "circt::esi";
  let description =
      [{"An interface for operations which carries channel semantics."}];

  let methods = [
    InterfaceMethod<
      [{"Returns the channel type of this operation."}],
      "circt::esi::ChannelType", "channelType"
    >,
    InterfaceMethod<
        [{"Returns the inner type of this channel. This will be the type of the
           data value of the channel, if the channel carries data semantics. Else,
           return NoneType."}],
        "mlir::Type",
        "innerType",
        (ins),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return $_op.channelType().getInner();
        }]>
  ];
}

//===----------------------------------------------------------------------===//
//
// Service-related interfaces.
//
//===----------------------------------------------------------------------===//

def ServiceDeclOpInterface : OpInterface<"ServiceDeclOpInterface"> {
  let cppNamespace = "circt::esi";
  let description = [{
    Any op which represents a service declaration should implement this
    interface.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the list of interface ports for this service interface.",
      "void", "getPortList",
      (ins "llvm::SmallVectorImpl<ServicePortInfo>&":$ports)
    >,
    InterfaceMethod<
        "Validate this service request against this service decl.",
        "mlir::LogicalResult", "validateRequest", (ins "mlir::Operation*":$req),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          return circt::esi::validateServiceConnectionRequest($_op, req);
        }]>,
  ];
}

def ServiceReqOpInterface : OpInterface<"ServiceReqOpInterface"> {
  let cppNamespace = "circt::esi";
  let description = [{
    Any op which is requesting connection to a service's port or has information
    pertaining to one of said requests.
  }];

  let methods = [
    InterfaceMethod<
      "Returns the service port symbol.",
      "hw::InnerRefAttr", "getServicePort", (ins)
    >,
    InterfaceMethod<
      "Returns the client name path.",
      "ArrayAttr", "getClientNamePath", (ins)
    >,
    InterfaceMethod<
      "Set the client name path.",
      "void", "setClientNamePath", (ins "ArrayAttr":$newClientNamePath),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        $_op.setClientNamePathAttr(newClientNamePath);
      }]>,
    InterfaceMethod<
      "Returns the type headed to the client. Can be null.",
      "Type", "getToClientType", (ins),
      /*methodBody=*/[{
        Value toClient = $_op.getToClient();
        if (!toClient)
          return {};
        return toClient.getType();
      }]>,
    InterfaceMethod<
      "Returns the value headed to the client. Can be null.",
      "Value", "getToClient", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]>,
    InterfaceMethod<
      "Returns the type headed to the server. Can be null.",
      "Type", "getToServerType", (ins),
      /*methodBody=*/[{
        Value toServer = $_op.getToServer();
        if (!toServer)
          return {};
        return toServer.getType();
      }]>,
    InterfaceMethod<
      "Returns the value headed to the server. Can be null.",
      "Value", "getToServer", (ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return {};
      }]>,
    InterfaceMethod<
      "Set the value headed to the server. Can assert().",
      "void", "setToServer", (ins "Value":$newValue),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        assert(false && "This op doesn't support the to server direction");
      }]>
  ];
}

#endif // CIRCT_DIALECT_ESI_INTERFACES_TD
