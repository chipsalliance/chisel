#include "PassDetails.h"

using namespace circt;
using namespace msft;

//===----------------------------------------------------------------------===//
// Common utilities for wire and partitioning passes
//===----------------------------------------------------------------------===//

namespace circt {
namespace msft {

struct MSFTPassCommon : PassCommon {
protected:
  /// Update all the instantiations of 'mod' to match the port list. For any
  /// output ports which survived, automatically map the result according to
  /// `newToOldResultMap`. Calls 'getOperandsFunc' with the new instance op, the
  /// old instance op, and expects the operand vector to return filled.
  /// `getOperandsFunc` can (and often does) modify other operations. The update
  /// call deletes the original instance op, so all references are invalidated
  /// after this call.
  SmallVector<InstanceOp, 1> updateInstances(
      MSFTModuleOp mod, ArrayRef<unsigned> newToOldResultMap,
      llvm::function_ref<void(InstanceOp, InstanceOp, SmallVectorImpl<Value> &)>
          getOperandsFunc);

  void getAndSortModules(ModuleOp topMod, SmallVectorImpl<MSFTModuleOp> &mods);

  void bubbleWiresUp(MSFTModuleOp mod);
  void dedupOutputs(MSFTModuleOp mod);
  void sinkWiresDown(MSFTModuleOp mod);
  void dedupInputs(MSFTModuleOp mod);
};

} // namespace msft
} // namespace circt
