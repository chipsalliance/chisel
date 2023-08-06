//===- ESIStdServices.cpp - ESI standard services -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIServices.h"
#include "circt/Dialect/ESI/ESITypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

#include <map>
#include <memory>

using namespace circt;
using namespace circt::esi;

/// Wrap types in esi channels and return the port info struct.
static ServicePortInfo createPort(StringRef name, Type toServerInner,
                                  Type toClientInner) {
  assert(toServerInner || toClientInner);
  auto *ctxt =
      toServerInner ? toServerInner.getContext() : toClientInner.getContext();
  return {StringAttr::get(ctxt, name), ChannelType::get(ctxt, toServerInner),
          ChannelType::get(ctxt, toClientInner)};
}

void RandomAccessMemoryDeclOp::getPortList(
    SmallVectorImpl<ServicePortInfo> &ports) {
  auto *ctxt = getContext();
  auto addressType = IntegerType::get(ctxt, llvm::Log2_64_Ceil(getDepth()));

  // Write port
  hw::StructType writeType = hw::StructType::get(
      ctxt,
      {hw::StructType::FieldInfo{StringAttr::get(ctxt, "address"), addressType},
       hw::StructType::FieldInfo{StringAttr::get(ctxt, "data"),
                                 getInnerType()}});
  ports.push_back(createPort("write", writeType, IntegerType::get(ctxt, 0)));

  // Read port
  ports.push_back(createPort("read", addressType, getInnerType()));
}
