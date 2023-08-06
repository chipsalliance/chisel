#include "circel/Bindings/Support/Context.h"
#include <atomic>

using namespace mlir::bindings;

Context::Context() : mlir(std::make_shared<mlir::MLIRContext>()) {
  static std::atomic_uint64_t nextID{0};
#ifdef MLIR_BINDINGS_USE_SINGULAR_IMMORTAL_CONTEXT
  assert(nextID++ == 0);
#else
  id = nextID++;
#endif
}

Context::~Context() {
#ifdef MLIR_BINDINGS_USE_SINGULAR_IMMORTAL_CONTEXT
  // Immortal contexts can't die
  assert(false);
#else
#endif
}