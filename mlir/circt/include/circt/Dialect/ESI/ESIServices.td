//===- ESIServices.td - ESI services ops -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ESI_SERVICES_TD
#define CIRCT_DIALECT_ESI_SERVICES_TD

include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "circt/Dialect/ESI/ESIDialect.td"
include "circt/Dialect/ESI/ESIInterfaces.td"
include "circt/Dialect/ESI/ESITypes.td"

include "circt/Dialect/HW/HWAttributesNaming.td"
include "circt/Dialect/HW/HWTypes.td"

def CustomServiceDeclOp : ESI_Op<"service.decl",
          [SingleBlock, NoTerminator, HasParent<"::mlir::ModuleOp">,
          Symbol, DeclareOpInterfaceMethods<ServiceDeclOpInterface>]> {
  let summary = "An ESI service interface declaration";
  let description = [{
    A declaration of an ESI service interface. Defines a contract between a
    service provider and its clients.

    Example:

    ```mlir
    esi.service.decl @HostComms {
      esi.service.to_server send : !esi.channel<!esi.any>
      esi.service.to_client recieve : !esi.channel<i8>
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region SizedRegion<1>:$ports);

  let assemblyFormat = [{
    $sym_name $ports attr-dict
  }];
}

def ToServerOp : ESI_Op<"service.to_server",
                        [HasParent<"::circt::esi::CustomServiceDeclOp">]> {
  let summary = "An ESI service port headed to the service";

  let arguments = (ins SymbolNameAttr:$inner_sym, TypeAttr:$toServerType);
  let assemblyFormat = [{
    $inner_sym attr-dict `:` $toServerType
  }];
}

def ToClientOp : ESI_Op<"service.to_client",
                        [HasParent<"::circt::esi::CustomServiceDeclOp">]> {
  let summary = "An ESI service port headed to a particular client";

  let arguments = (ins SymbolNameAttr:$inner_sym, TypeAttr:$toClientType);
  let assemblyFormat = [{
    $inner_sym  attr-dict `:` $toClientType
  }];
}

def ServiceDeclInOutOp : ESI_Op<"service.inout",
                        [HasParent<"::circt::esi::CustomServiceDeclOp">]> {
  let summary = "An ESI service port which has both directions";

  let arguments = (ins SymbolNameAttr:$inner_sym,
                       TypeAttr:$toServerType, TypeAttr:$toClientType);
  let assemblyFormat = [{
    $inner_sym attr-dict
    `:` qualified($toServerType) `->` qualified($toClientType)
  }];
}

def ESIAnyType : ESI_Type<"Any"> {
  let summary = "any type";
  let description = [{
    Used to state that any type is accepted in a service port declaration. The
    specific type will be determined later in compilation.
  }];

  let mnemonic = "any";
  let assemblyFormat = "";

  let extraClassDeclaration = [{
    static AnyType get(MLIRContext *context);
  }];
}

def ServiceInstanceOp : ESI_Op<"service.instance"> {
  let summary = "Instantiate a server module";
  let description = [{
    Instantiate a service adhering to a service declaration interface.

    A pass collects all of the connection requests to the service this op
    implements from the containing modules' descendants (in the instance
    hierarchy). It bubbles them all up to the module containing this op,
    creating the necessary ESI channel ports, groups them appropriately, then
    replaces this op with a `service.impl_req`.

    If 'service_symbol' isn't specified, this instance will be used to implement
    all of the service requests which get surfaced to here. This option is
    generally used at the top level to specify host connectivity.

    Since implementing the server will usually need "normal" I/O, `inputs` and
    `results` act like normal `hw.instance` ports.

    $identifier is used by frontends to specify or remember the type of
    implementation to use for this service.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$service_symbol,
                       StrAttr:$impl_type,
                       OptionalAttr<DictionaryAttr>:$impl_opts,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    (`svc` $service_symbol^)? `impl` `as` $impl_type (`opts` $impl_opts^)?
      `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];
}

def ServiceImplementReqOp : ESI_Op<"service.impl_req", [NoTerminator]> {
  let summary = "Request for a service to be implemented";
  let description = [{
    The connect services pass replaces `service.instance`s with this op. The
    `portReqs` region is the set of connection requests which need to be
    implemented for this service instance. Channels to/from the requests have
    been added to the operands/results of this op and consumers/producers have
    been redirected.

    Some other pass or frontend is expected to replace this op with an actual
    implementation.
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$service_symbol,
                       StrAttr:$impl_type,
                       OptionalAttr<DictionaryAttr>:$impl_opts,
                       Variadic<AnyType>:$inputs);
  let results = (outs Variadic<AnyType>:$outputs);
  let regions = (region SizedRegion<1>:$portReqs);

  let assemblyFormat = [{
    (`svc` $service_symbol^)? `impl` `as` $impl_type (`opts` $impl_opts^)?
     `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
      $portReqs
  }];
}

def RequestToServerConnectionOp : ESI_Op<"service.req.to_server", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        ServiceReqOpInterface]> {
  let summary = "Request a connection to send data";

  let arguments = (ins InnerRefAttr:$servicePort,
                       ChannelType:$toServer, StrArrayAttr:$clientNamePath);
  let assemblyFormat = [{
    $toServer `->` $servicePort `(` $clientNamePath `)`
      attr-dict `:` qualified(type($toServer))
  }];

  let extraClassDeclaration = [{
    // Set the value headed to the server. Overrides method in
    // ServiceReqOpInterface.
    void setToServer(Value v) {
      getToServerMutable().assign(v);
    }
  }];
}

def RequestToClientConnectionOp : ESI_Op<"service.req.to_client", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        ServiceReqOpInterface]> {
  let summary = "Request a connection to receive data";

  let arguments = (ins InnerRefAttr:$servicePort,
                       StrArrayAttr:$clientNamePath);
  let results = (outs ChannelType:$toClient);
  let assemblyFormat = [{
    $servicePort `(` $clientNamePath `)`
      attr-dict `:` qualified(type($toClient))
  }];
}

def RequestInOutChannelOp : ESI_Op<"service.req.inout", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>,
        ServiceReqOpInterface]> {
  let summary = "Request a bidirectional channel";

  let arguments = (ins InnerRefAttr:$servicePort,
                       ChannelType:$toServer,
                       StrArrayAttr:$clientNamePath);
  let results = (outs ChannelType:$toClient);

  let assemblyFormat = [{
    $toServer `->` $servicePort `(` $clientNamePath `)` attr-dict `:`
      qualified(type($toServer)) `->` qualified(type($toClient))
  }];

  let extraClassDeclaration = [{
    // Set the value headed to the server. Overrides method in
    // ServiceReqOpInterface.
    void setToServer(Value v) {
      getToServerMutable().assign(v);
    }
  }];
}

def ServiceHierarchyMetadataOp : ESI_Op<"service.hierarchy.metadata", [
        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "Metadata about a service in the service hierarchy";

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$service_symbol,
                       ArrayAttr:$serverNamePath,
                       StrAttr:$impl_type,
                       OptionalAttr<DictionaryAttr>:$impl_details,
                       ArrayAttr:$clients);
  let assemblyFormat = [{
    `path` $serverNamePath (`implementing` $service_symbol^)?
      `impl` `as` $impl_type (`opts` $impl_details^)?
      `clients` $clients attr-dict 
  }];
}

#endif // CIRCT_DIALECT_ESI_SERVICES_TD
