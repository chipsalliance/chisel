//===- GeneratorCallout.cpp - Generator Callout Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Call arbitrary programs and pass them the attributes attached to external
// modules.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

using namespace circt;
using namespace sv;
using namespace hw;

//===----------------------------------------------------------------------===//
// GeneratorCalloutPass
//===----------------------------------------------------------------------===//

namespace {

struct HWGeneratorCalloutPass
    : public sv::HWGeneratorCalloutPassBase<HWGeneratorCalloutPass> {
  void runOnOperation() override;

  void processGenerator(HWModuleGeneratedOp generatedModuleOp,
                        StringRef generatorExe,
                        ArrayRef<StringRef> extraGeneratorArgs);
};
} // end anonymous namespace

void HWGeneratorCalloutPass::runOnOperation() {
  ModuleOp root = getOperation();
  SmallVector<StringRef> genOptions;
  StringRef extraGeneratorArgs(genExecArgs);
  extraGeneratorArgs.split(genOptions, ';');

  SmallString<32> execName = llvm::sys::path::filename(genExecutable);
  SmallString<32> execPath = llvm::sys::path::parent_path(genExecutable);

  auto generatorExe = llvm::sys::findProgramByName(execName, {execPath});
  // If program not found, search it in $PATH.
  if (!generatorExe)
    generatorExe = llvm::sys::findProgramByName(execName);
  // If cannot find the executable, then nothing to do, return.
  if (!generatorExe) {
    root.emitError("cannot find executable '" + execName + "' in path '" +
                   execPath + "'");
    return;
  }
  for (auto &op : llvm::make_early_inc_range(root.getBody()->getOperations())) {
    if (auto generator = dyn_cast<HWModuleGeneratedOp>(op))
      processGenerator(generator, *generatorExe, extraGeneratorArgs);
  }
}

void HWGeneratorCalloutPass::processGenerator(
    HWModuleGeneratedOp generatedModuleOp, StringRef generatorExe,
    ArrayRef<StringRef> extraGeneratorArgs) {
  // Get the corresponding schema associated with this generated op.
  auto genSchema =
      dyn_cast<HWGeneratorSchemaOp>(generatedModuleOp.getGeneratorKindOp());
  if (!genSchema)
    return;

  // Ignore the generator op if the schema does not match the user specified
  // schema name from command line "-schema-name"
  if (genSchema.getDescriptor().str() != schemaName)
    return;

  SmallVector<std::string> generatorArgs;
  // First argument should be the executable name.
  generatorArgs.push_back(generatorExe.str());
  for (auto o : extraGeneratorArgs)
    generatorArgs.push_back(o.str());

  auto moduleName =
      generatedModuleOp.getVerilogModuleNameAttr().getValue().str();
  // The moduleName option is not present in the schema, so add it
  // explicitly.
  generatorArgs.push_back("--moduleName");
  generatorArgs.push_back(moduleName);
  // Iterate over all the attributes in the schema.
  // Assumption: All the options required by the generator program must be
  // present in the schema.
  for (auto attr : genSchema.getRequiredAttrs()) {
    auto sAttr = attr.cast<StringAttr>();
    // Get the port name from schema.
    StringRef portName = sAttr.getValue();
    generatorArgs.push_back("--" + portName.str());
    // Get the value for the corresponding port name.
    auto v = generatedModuleOp->getAttr(portName);
    if (auto intV = v.dyn_cast<IntegerAttr>())
      generatorArgs.push_back(std::to_string(intV.getValue().getZExtValue()));
    else if (auto strV = v.dyn_cast<StringAttr>())
      generatorArgs.push_back(strV.getValue().str());
    else {
      generatedModuleOp.emitError(
          "portname attribute " + portName +
          " value specified on the rtl.module.generated operation is not "
          "handled, "
          "only integer and string types supported.");
      return;
    }
  }
  SmallVector<StringRef> generatorArgStrRef;
  for (const std::string &a : generatorArgs)
    generatorArgStrRef.push_back(a);

  std::string errMsg;
  SmallString<32> genExecOutFileName;
  auto errCode = llvm::sys::fs::getPotentiallyUniqueTempFileName(
      "generatorCalloutTemp", StringRef(""), genExecOutFileName);
  // Default error code is 0.
  std::error_code ok;
  if (errCode != ok) {
    generatedModuleOp.emitError("cannot generate a unique temporary file name");
    return;
  }
  std::optional<StringRef> redirects[] = {
      std::nullopt, StringRef(genExecOutFileName), std::nullopt};
  int result = llvm::sys::ExecuteAndWait(
      generatorExe, generatorArgStrRef, /*Env=*/std::nullopt,
      /*Redirects=*/redirects,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

  if (result != 0) {
    generatedModuleOp.emitError("execution of '" + generatorExe + "' failed");
    return;
  }

  auto bufferRead = llvm::MemoryBuffer::getFile(genExecOutFileName);
  if (!bufferRead || !*bufferRead) {
    generatedModuleOp.emitError("execution of '" + generatorExe +
                                "' did not produce any output file named '" +
                                genExecOutFileName + "'");
    return;
  }

  // Only extract the first line from the output.
  auto fileContent = (*bufferRead)->getBuffer().split('\n').first.str();
  OpBuilder builder(generatedModuleOp);
  auto extMod = builder.create<hw::HWModuleExternOp>(
      generatedModuleOp.getLoc(), generatedModuleOp.getVerilogModuleNameAttr(),
      generatedModuleOp.getPortList());
  // Attach an attribute to which file the definition of the external
  // module exists in.
  extMod->setAttr("filenames", builder.getStringAttr(fileContent));
  generatedModuleOp.erase();
}

std::unique_ptr<Pass> circt::sv::createHWGeneratorCalloutPass() {
  return std::make_unique<HWGeneratorCalloutPass>();
}
