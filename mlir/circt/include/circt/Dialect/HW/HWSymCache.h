//===- HWSymCache.h - Declare Symbol Cache ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares a Symbol Cache specialized for HW instances.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_SYMCACHE_H
#define CIRCT_DIALECT_HW_SYMCACHE_H

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/SymCache.h"

namespace circt {
namespace hw {

/// This stores lookup tables to make manipulating and working with the IR more
/// efficient.  There are two phases to this object: the "building" phase in
/// which it is "write only" and then the "using" phase which is read-only (and
/// thus can be used by multiple threads).  The  "freeze" method transitions
/// between the two states.
class HWSymbolCache : public SymbolCacheBase {
public:
  class Item {
  public:
    Item(mlir::Operation *op) : op(op), port(~0ULL) {}
    Item(mlir::Operation *op, size_t port) : op(op), port(port) {}
    bool hasPort() const { return port != ~0ULL; }
    size_t getPort() const { return port; }
    mlir::Operation *getOp() const { return op; }

  private:
    mlir::Operation *op;
    size_t port;
  };

  // Add inner names, which might be ports
  void addDefinition(mlir::StringAttr modSymbol, mlir::StringAttr name,
                     mlir::Operation *op, size_t port = ~0ULL) {
    auto key = InnerRefAttr::get(modSymbol, name);
    symbolCache.try_emplace(key, op, port);
  }

  void addDefinition(mlir::Attribute key, mlir::Operation *op) override {
    assert(!isFrozen && "cannot mutate a frozen cache");
    symbolCache.try_emplace(key, op);
  }

  // Pull in getDefinition(mlir::FlatSymbolRefAttr symbol)
  using SymbolCacheBase::getDefinition;
  mlir::Operation *getDefinition(mlir::Attribute attr) const override {
    assert(isFrozen && "cannot read from this cache until it is frozen");
    auto it = symbolCache.find(attr);
    if (it == symbolCache.end())
      return nullptr;
    assert(!it->second.hasPort() && "Module names should never be ports");
    return it->second.getOp();
  }

  HWSymbolCache::Item getInnerDefinition(mlir::StringAttr modSymbol,
                                         mlir::StringAttr name) const {
    return lookupInner(InnerRefAttr::get(modSymbol, name));
  }

  HWSymbolCache::Item getInnerDefinition(InnerRefAttr inner) const {
    return lookupInner(inner);
  }

  /// Mark the cache as frozen, which allows it to be shared across threads.
  void freeze() { isFrozen = true; }

private:
  Item lookupInner(InnerRefAttr attr) const {
    assert(isFrozen && "cannot read from this cache until it is frozen");
    auto it = symbolCache.find(attr);
    return it == symbolCache.end() ? Item{nullptr, ~0ULL} : it->second;
  }

  bool isFrozen = false;

  /// This stores a lookup table from symbol attribute to the item
  /// that defines it.
  llvm::DenseMap<mlir::Attribute, Item> symbolCache;

private:
  // Iterator support. Map from Item's to their inner operations.
  using Iterator = decltype(symbolCache)::iterator;
  struct HwSymbolCacheIteratorImpl : public CacheIteratorImpl {
    HwSymbolCacheIteratorImpl(Iterator it) : it(it) {}
    CacheItem operator*() override {
      return {it->getFirst(), it->getSecond().getOp()};
    }
    void operator++() override { it++; }
    bool operator==(CacheIteratorImpl *other) override {
      return it == static_cast<HwSymbolCacheIteratorImpl *>(other)->it;
    }
    Iterator it;
  };

public:
  SymbolCacheBase::Iterator begin() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<HwSymbolCacheIteratorImpl>(symbolCache.begin()));
  }
  SymbolCacheBase::Iterator end() override {
    return SymbolCacheBase::Iterator(
        std::make_unique<HwSymbolCacheIteratorImpl>(symbolCache.end()));
  }
};

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_SYMCACHE_H
