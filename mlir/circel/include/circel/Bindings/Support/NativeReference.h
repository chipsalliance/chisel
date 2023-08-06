#ifndef MLIR_BINDINGS_SUPPORT_NATIVE_REFERENCE_H_
#define MLIR_BINDINGS_SUPPORT_NATIVE_REFERENCE_H_

#include <atomic>
#include <cassert>
#include <mlir/IR/Types.h>
#include <mlir/Support/TypeID.h>

namespace mlir {
namespace bindings {

class AnyNativeReference {
  class AnyStorage {
    friend class AnyNativeReference;

    std::atomic_uint64_t referenceCount = 1;
    mlir::TypeID typeID;
    using StoredDestructor = void (*)(AnyStorage *storage);
    StoredDestructor storedDestructor;

  protected:
    AnyStorage(mlir::TypeID typeID, StoredDestructor storedDestructor)
        : typeID(typeID), storedDestructor(storedDestructor) {}
  };
  AnyStorage *storage;

protected:
  template <typename T> class Storage : public AnyStorage {
    friend class AnyNativeReference;
    T value;

  public:
    template <typename... Args>
    Storage(Args &&...args)
        : AnyStorage(mlir::TypeID::get<T>(),
                     [](AnyStorage *anyStorage) {
                       auto storage = static_cast<Storage<T> *>(anyStorage);
                       delete storage;
                     }),
          value(std::forward<Args>(args)...) {}
  };

  explicit AnyNativeReference(AnyStorage *storage) : storage(storage) {}

  explicit AnyNativeReference(void *opaqueReference)
      : storage(static_cast<AnyStorage *>(opaqueReference)) {
    // Increment reference count to balance the decrement when this value is
    // destroyed.
    storage->referenceCount++;
  }

  template <typename T> void assertStorageTypeMatches() {
    assert(mlir::TypeID::get<T>() == storage->typeID);
  }

  template <typename T> const T &getValue() {
    assertStorageTypeMatches<T>();
    return static_cast<Storage<T> *>(storage)->value;
  }

public:
  AnyNativeReference(const AnyNativeReference &source)
      : storage(source.storage) {
    storage->referenceCount++;
  }
  ~AnyNativeReference() {
    if (--storage->referenceCount == 0)
      storage->storedDestructor(storage);
  }

  void *getRetainedOpaqueReference() {
    storage->referenceCount++;
    return reinterpret_cast<void *>(storage);
  }

  static void releaseOpaqueReference(void *opaqueReference) {
    auto reference = AnyNativeReference(opaqueReference);
    // Consume the unbalanced retain from `getRetainedOpaqueReference`, which
    // cannot be the last retain because creating `reference` incremented the
    // retain count, which will be decremented when `reference` is destroyed.
    assert(reference.storage->referenceCount-- > 0);
  }
};
template <typename T> class NativeReference : public AnyNativeReference {
  NativeReference(Storage<T> *storage) : AnyNativeReference(storage) {}

  NativeReference(void *opaqueReference)
      : AnyNativeReference(opaqueReference) {}

public:
  NativeReference(const NativeReference &source) : AnyNativeReference(source) {}

  template <typename... Args> static NativeReference<T> create(Args &&...args) {
    return NativeReference<T>(new Storage<T>(std::forward<Args>(args)...));
  }

  static NativeReference<T> getFromOpaqueReference(void *opaqueReference) {
    auto reference = NativeReference<T>(opaqueReference);
    reference.template assertStorageTypeMatches<T>();
    return reference;
  }

  const T &getValue() { return AnyNativeReference::getValue<T>(); }
};

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_NATIVE_REFERENCE_H_
