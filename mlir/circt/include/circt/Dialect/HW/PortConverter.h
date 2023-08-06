//===- PortConverter.h - Module I/O rewriting utility -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The PortConverter is a utility class for rewriting arguments of a
// HWMutableModuleLike operation.
// It is intended to be a generic utility that can facilitate replacement of
// a given module in- or output to an arbitrary set of new inputs and outputs
// (i.e. 1 port -> N in, M out ports). Typical usecases is where an in (or
// output) of a module represents some higher-level abstraction that will be
// implemented by a set of lower-level in- and outputs ports + supporting
// operations within a module. It also attempts to do so in an optimal way, by
// e.g. being able to collect multiple port modifications of a module, and
// perform them all at once.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_PORTCONVERTER_H
#define CIRCT_DIALECT_HW_PORTCONVERTER_H

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/LLVM.h"

namespace circt {
namespace hw {

class PortConversionBuilder;
class PortConversion;

class PortConverterImpl {
public:
  /// Run port conversion.
  LogicalResult run();
  Block *getBody() const { return body; }
  hw::HWMutableModuleLike getModule() const { return mod; }

  /// These two methods take care of allocating new ports in the correct place
  /// based on the position of 'origPort'. The new port is based on the original
  /// name and suffix. The specification for the new port is given by `newPort`
  /// and is recorded internally. Any changes to 'newPort' after calling this
  /// will not be reflected in the modules new port list. Will also add the new
  /// input to the block arguments of the body of the module.
  Value createNewInput(hw::PortInfo origPort, const Twine &suffix, Type type,
                       hw::PortInfo &newPort);
  /// Same as above. 'output' is the value fed into the new port and is required
  /// if 'body' is non-null. Important note: cannot be a backedge which gets
  /// replaced since this isn't attached to an op until later in the pass.
  void createNewOutput(hw::PortInfo origPort, const Twine &suffix, Type type,
                       Value output, hw::PortInfo &newPort);

protected:
  PortConverterImpl(hw::InstanceGraphNode *moduleNode)
      : moduleNode(moduleNode) {
    mod = dyn_cast<hw::HWMutableModuleLike>(*moduleNode->getModule());
    assert(mod && "PortConverter only works on HWMutableModuleLike");

    if (mod->getNumRegions() == 1 && mod->getRegion(0).hasOneBlock())
      body = &mod->getRegion(0).front();
  }

  std::unique_ptr<PortConversionBuilder> ssb;

private:
  /// Materializes/commits all of the recorded port changes to the module.
  void materializeChanges();

  /// Updates an instance of the module. This is called after the module has
  /// been updated. It will update the instance to match the new port
  void updateInstance(hw::InstanceOp);

  // If the module has a block and it wants to be modified, this'll be
  // non-null.
  Block *body = nullptr;

  hw::InstanceGraphNode *moduleNode;
  hw::HWMutableModuleLike mod;

  // Keep around a reference to the specific port conversion classes to
  // facilitate updating the instance ops. Indexed by the original port
  // location.
  SmallVector<std::unique_ptr<PortConversion>> loweredInputs;
  SmallVector<std::unique_ptr<PortConversion>> loweredOutputs;

  // Tracking information to modify the module. Populated by the
  // 'createNew(Input|Output)' methods. Will be cleared once port changes have
  // materialized. Default length is  0 to save memory in case we'll be keeping
  // this around for later use.
  SmallVector<std::pair<unsigned, hw::PortInfo>, 0> newInputs;
  SmallVector<std::pair<unsigned, hw::PortInfo>, 0> newOutputs;
  SmallVector<Value, 0> newOutputValues;
};

/// Base class for the port conversion of a particular port. Abstracts the
/// details of a particular port conversion from the port layout. Subclasses
/// keep around port mapping information to use when updating instances.
class PortConversion {
public:
  PortConversion(PortConverterImpl &converter, hw::PortInfo origPort)
      : converter(converter), body(converter.getBody()), origPort(origPort) {}
  virtual ~PortConversion() = default;

  // An optional initialization step that can be overridden by subclasses.
  // This allows subclasses to perform a failable post-construction
  // initialization step.
  virtual LogicalResult init() { return success(); }

  // Lower the specified port into a wire-level signaling protocol. The two
  // virtual methods 'build*Signals' should be overridden by subclasses. They
  // should use the 'create*' methods in 'PortConverter' to create the
  // necessary ports.
  void lowerPort() {
    if (origPort.dir == hw::ModulePort::Direction::Output)
      buildOutputSignals();
    else
      buildInputSignals();
  }

  /// Update an instance port to the new port information.
  virtual void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                               SmallVectorImpl<Value> &newOperands,
                               ArrayRef<Backedge> newResults) = 0;
  virtual void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                                SmallVectorImpl<Value> &newOperands,
                                ArrayRef<Backedge> newResults) = 0;

  MLIRContext *getContext() { return getModule()->getContext(); }
  bool isUntouched() const { return isUntouchedFlag; }

protected:
  // Build the input and output signals for the port. This pertains to modifying
  // the module itself.
  virtual void buildInputSignals() = 0;
  virtual void buildOutputSignals() = 0;

  PortConverterImpl &converter;
  Block *body;
  hw::PortInfo origPort;

  hw::HWMutableModuleLike getModule() { return converter.getModule(); }

  // We don't need full LLVM-style RTTI support for PortConversion (would
  // require some mechanism of registering user-provided PortConversion-derived
  // classes), we only need to dynamically tell whether any given PortConversion
  // is the UntouchedPortConversion.
  bool isUntouchedFlag = false;
}; // namespace hw

// A PortConversionBuilder will, given an input type, build the appropriate
// port conversion for that type.
class PortConversionBuilder {
public:
  PortConversionBuilder(PortConverterImpl &converter) : converter(converter) {}
  virtual ~PortConversionBuilder() = default;

  // Builds the appropriate port conversion for the port. Users should
  // override this method with their own llvm::TypeSwitch-based dispatch code,
  // and by default call this method when no port conversion applies.
  virtual FailureOr<std::unique_ptr<PortConversion>> build(hw::PortInfo port);

  PortConverterImpl &converter;
};

// A PortConverter wraps a single HWMutableModuleLike operation, and is
// initialized from an instance graph node. The port converter is templated
// on a PortConversionBuilder, which is used to build the appropriate
// port conversion for each port type.
template <typename PortConversionBuilderImpl>
class PortConverter : public PortConverterImpl {
public:
  PortConverter(hw::InstanceGraph &graph, hw::HWMutableModuleLike mod)
      : PortConverterImpl(graph.lookup(cast<hw::HWModuleLike>(*mod))) {
    ssb = std::make_unique<PortConversionBuilderImpl>(*this);
  }
};

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_PORTCONVERTER_H
