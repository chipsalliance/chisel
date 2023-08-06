#ifndef MLIR_BINDINGS_SUPPORT_CONTEXT_H_
#define MLIR_BINDINGS_SUPPORT_CONTEXT_H_

#include <memory>
#include <mlir/IR/MLIRContext.h>

namespace mlir {
namespace bindings {

class Context {
  friend class AnyContextual;
  std::shared_ptr<mlir::MLIRContext> mlir;
#ifndef MLIR_BINDINGS_USE_SINGULAR_IMMORTAL_CONTEXT
  uint64_t id;
#endif

public:
  mlir::MLIRContext *getMLIR() const { return mlir.get(); }

  Context();
  ~Context();
};

class AnyContextual {
#ifndef MLIR_BINDINGS_USE_SINGULAR_IMMORTAL_CONTEXT
  uint64_t contextID;
#endif

protected:
  AnyContextual(const Context &context) { contextID = context.id; }
  void assertUnwrappableInContext(const Context &context) const {
#ifndef MLIR_BINDINGS_USE_SINGULAR_IMMORTAL_CONTEXT
    assert(contextID == context.id);
#endif
  }
};
template <typename T> class Contextual : public AnyContextual {
  T value;

public:
  Contextual<T>(const Context &context, T value)
      : AnyContextual(context), value(value) {}

  /**
   * Get the value, verifying that it is in the given context.
   * Most contextual values are simple pointers into context-owned memory, so
   * returning them via copy is OK.
   */
  T getValueInContext(const Context &context) const {
    assertUnwrappableInContext(context);
    return value;
  }

  /**
   * Get the value without verifying the context. This is an escape hatch and
   * should only be used for debugging purposes.
   */
  T getUnsafeValue() const { return value; }
};

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_CONTEXT_H_
