//===- ResolveTraces.cpp - Resolve TraceAnnotations -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Find any TraceAnnotations in the design, update their targets, and write the
// annotations out to an output annotation file.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

#define DEBUG_TYPE "firrtl-resolve-traces"

using namespace circt;
using namespace firrtl;

/// Expand a TraceNameAnnotation (which has don't touch semantics) into a
/// TraceAnnotation (which does NOT have don't touch semantics) and separate
/// DontTouchAnnotations for targets that are not modules, external modules, or
/// instances (as these targets are not valid for a don't touch).
LogicalResult circt::firrtl::applyTraceName(const AnnoPathValue &target,
                                            DictionaryAttr anno,
                                            ApplyState &state) {

  auto *context = anno.getContext();

  NamedAttrList trace, dontTouch;
  for (auto namedAttr : anno.getValue()) {
    if (namedAttr.getName() == "class") {
      trace.append("class", StringAttr::get(context, traceAnnoClass));
      dontTouch.append("class", StringAttr::get(context, dontTouchAnnoClass));
      continue;
    }
    trace.append(namedAttr);

    // When we see the "target", check to see if this is not targeting a module,
    // extmodule, or instance (as these are invalid "don't touch" targets).  If
    // it is not, then add a DontTouchAnnotation.
    if (namedAttr.getName() == "target" &&
        !target.isOpOfType<FModuleOp, FExtModuleOp, InstanceOp>()) {
      dontTouch.append(namedAttr);
      state.addToWorklistFn(DictionaryAttr::getWithSorted(context, dontTouch));
    }
  }

  state.addToWorklistFn(DictionaryAttr::getWithSorted(context, trace));

  return success();
}

struct ResolveTracesPass : public ResolveTracesBase<ResolveTracesPass> {
  using ResolveTracesBase::outputAnnotationFilename;

  void runOnOperation() override;

private:
  /// Stores a pointer to an NLA Table.  This is populated during
  /// runOnOperation.
  NLATable *nlaTable;

  /// Stores a pointer to an inner symbol table collection.
  hw::InnerSymbolTableCollection *istc;

  /// Global symbol index used for substitutions, e.g., "{{42}}".  This value is
  /// the _next_ index that will be used.
  unsigned symbolIdx = 0;

  /// Map of symbol to symbol index.  This is used to reuse symbol
  /// substitutions.
  DenseMap<Attribute, unsigned> symbolMap;

  /// Symbol substitutions for the JSON verbatim op.
  SmallVector<Attribute> symbols;

  /// Get a symbol index and update symbol datastructures.
  unsigned getSymbolIndex(Attribute attr) {
    auto iterator = symbolMap.find(attr);
    if (iterator != symbolMap.end())
      return iterator->getSecond();

    auto idx = symbolIdx++;
    symbolMap.insert({attr, idx});
    symbols.push_back(attr);

    return idx;
  }

  /// Convert an annotation path to a string with symbol substitutions.
  void buildTarget(AnnoPathValue &path, SmallString<64> &newTarget) {

    auto addSymbol = [&](Attribute attr) -> void {
      newTarget.append("{{");
      Twine(getSymbolIndex(attr)).toVector(newTarget);
      newTarget.append("}}");
    };

    newTarget.append("~");
    newTarget.append(
        path.ref.getModule()->getParentOfType<CircuitOp>().getName());
    newTarget.append("|");

    if (path.isLocal()) {
      addSymbol(
          FlatSymbolRefAttr::get(path.ref.getModule().getModuleNameAttr()));
    } else {
      addSymbol(FlatSymbolRefAttr::get(path.instances.front()
                                           ->getParentOfType<FModuleLike>()
                                           .getModuleNameAttr()));
    }

    for (auto inst : path.instances) {
      newTarget.append("/");
      addSymbol(hw::InnerRefAttr::get(
          inst->getParentOfType<FModuleLike>().getModuleNameAttr(),
          inst.getInnerSymAttr().getSymName()));
      newTarget.append(":");
      addSymbol(inst.getModuleNameAttr());
    }

    // If this targets a module or an instance, then we're done.  There is no
    // "reference" part of the FIRRTL target.
    if (path.ref.isa<OpAnnoTarget>() &&
        path.isOpOfType<FModuleOp, FExtModuleOp, InstanceOp>())
      return;

    std::optional<ModuleNamespace> moduleNamespace;

    newTarget.append(">");
    auto innerSymStr =
        TypeSwitch<AnnoTarget, StringAttr>(path.ref)
            .Case<PortAnnoTarget>([&](PortAnnoTarget portTarget) {
              return hw::InnerSymbolTable::getInnerSymbol(hw::InnerSymTarget(
                  portTarget.getPortNo(), portTarget.getModule(), 0));
            })
            .Case<OpAnnoTarget>([&](OpAnnoTarget opTarget) {
              return hw::InnerSymbolTable::getInnerSymbol(opTarget.getOp());
            })
            .Default([](auto) {
              assert(false && "unexpected annotation target type");
              return StringAttr{};
            });
    addSymbol(hw::InnerRefAttr::get(path.ref.getModule().getModuleNameAttr(),
                                    innerSymStr));

    auto type = dyn_cast<FIRRTLBaseType>(path.ref.getType());
    assert(type && "expected a FIRRTLBaseType");
    auto targetFieldID = path.fieldIdx;
    while (targetFieldID) {
      FIRRTLTypeSwitch<FIRRTLBaseType>(type)
          .Case<FVectorType>([&](FVectorType vector) {
            auto index = vector.getIndexForFieldID(targetFieldID);
            newTarget.append("[");
            Twine(index).toVector(newTarget);
            newTarget.append("]");
            type = vector.getElementType();
            targetFieldID -= vector.getFieldID(index);
          })
          .template Case<BundleType>([&](BundleType bundle) {
            auto index = bundle.getIndexForFieldID(targetFieldID);
            newTarget.append(".");
            newTarget.append(bundle.getElementName(index));
            type = bundle.getElementType(index);
            targetFieldID -= bundle.getFieldID(index);
          })
          .Default([&](auto) { targetFieldID = 0; });
    }
  }

  /// Internal implementation that updates an Annotation to add a "target" field
  /// based on the current location of the annotation in the circuit.  The value
  /// of the "target" will be a local target if the Annotation is local and a
  /// non-local target if the Annotation is non-local.
  AnnoPathValue updateTargetImpl(Annotation &anno, FModuleLike &module,
                                 FIRRTLBaseType type, hw::InnerRefAttr name,
                                 AnnoTarget target) {
    SmallString<64> newTarget("~");
    newTarget.append(module->getParentOfType<CircuitOp>().getName());
    newTarget.append("|");

    SmallVector<InstanceOp> instances;

    if (auto nla = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
      hw::HierPathOp path = nlaTable->getNLA(nla.getAttr());
      for (auto part : path.getNamepath().getValue().drop_back()) {
        auto inst = cast<hw::InnerRefAttr>(part);
        instances.push_back(dyn_cast<InstanceOp>(
            istc->getInnerSymbolTable(nlaTable->getModule(inst.getModule()))
                .lookupOp(inst.getName())));
      }
    }

    AnnoPathValue path(instances, target, anno.getFieldID());

    return path;
  }

  /// Add a "target" field to a port Annotation that indicates the current
  /// location of the port in the circuit.
  std::optional<AnnoPathValue> updatePortTarget(FModuleLike &module,
                                                Annotation &anno,
                                                unsigned portIdx,
                                                hw::InnerRefAttr innerRef) {
    auto type = getBaseType(type_cast<FIRRTLType>(module.getPortType(portIdx)));
    return updateTargetImpl(anno, module, type, innerRef,
                            PortAnnoTarget(module, portIdx));
  }

  /// Add a "target" field to an Annotation that indicates the current location
  /// of a component in the circuit.
  std::optional<AnnoPathValue> updateTarget(FModuleLike &module, Operation *op,
                                            Annotation &anno,
                                            hw::InnerRefAttr innerRef) {

    // Get the type of the operation either by checking for the
    // result targeted by symbols on it (which are used to track the op)
    // or by inspecting its single result.
    auto is = dyn_cast<hw::InnerSymbolOpInterface>(op);
    Type type;
    if (is && is.getTargetResult())
      type = is.getTargetResult().getType();
    else {
      if (op->getNumResults() != 1)
        return std::nullopt;
      type = op->getResultTypes().front();
    }

    auto baseType = getBaseType(type_cast<FIRRTLType>(type));
    return updateTargetImpl(anno, module, baseType, innerRef, OpAnnoTarget(op));
  }

  /// Add a "target" field to an Annotation on a Module that indicates the
  /// current location of the module.  This will be local or non-local depending
  /// on the Annotation.
  std::optional<AnnoPathValue> updateModuleTarget(FModuleLike &module,
                                                  Annotation &anno) {
    SmallVector<InstanceOp> instances;

    if (auto nla = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
      hw::HierPathOp path = nlaTable->getNLA(nla.getAttr());
      for (auto part : path.getNamepath().getValue().drop_back()) {
        auto inst = cast<hw::InnerRefAttr>(part);
        instances.push_back(cast<InstanceOp>(
            istc->getInnerSymbolTable(nlaTable->getModule(inst.getModule()))
                .lookupOp(inst.getName())));
      }
    }

    AnnoPathValue path(instances, OpAnnoTarget(module), 0);

    return path;
  }
};

void ResolveTracesPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running ResolveTraces "
                      "-----------------------------------------------===\n"
                   << "Annotation Modifications:\n");

  // Grab the circuit (as this is used a few times below).
  CircuitOp circuit = getOperation();
  MLIRContext *context = circuit.getContext();

  // Populate pointer datastructures.
  nlaTable = &getAnalysis<NLATable>();
  istc = &getAnalysis<hw::InnerSymbolTableCollection>();

  // Function to find all Trace Annotations in the circuit, add a "target" field
  // to them indicating the current local/non-local target of the operation/port
  // the Annotation is attached to, copy the annotation into an
  // "outputAnnotations" return vector, and delete the original Annotation.  If
  // a component or port is targeted by a Trace Annotation it will be given a
  // symbol to prevent the output Trace Annotation from being made invalid by a
  // later optimization.
  auto onModule = [&](FModuleLike moduleLike) {
    // Output Trace Annotations from this module only.
    SmallVector<std::pair<Annotation, AnnoPathValue>> outputAnnotations;

    // A lazily constructed module namespace.
    std::optional<ModuleNamespace> moduleNamespace;

    // Return a cached module namespace, lazily constructing it if needed.
    auto getNamespace = [&](FModuleLike module) -> ModuleNamespace & {
      if (!moduleNamespace)
        moduleNamespace = ModuleNamespace(module);
      return *moduleNamespace;
    };

    // Visit the module.
    AnnotationSet::removeAnnotations(moduleLike, [&](Annotation anno) {
      if (!anno.isClass(traceAnnoClass))
        return false;

      auto path = updateModuleTarget(moduleLike, anno);
      if (!path)
        return false;

      outputAnnotations.push_back({anno, *path});
      return true;
    });

    // Visit port annotations.
    AnnotationSet::removePortAnnotations(
        moduleLike, [&](unsigned portIdx, Annotation anno) {
          if (!anno.isClass(traceAnnoClass))
            return false;

          hw::InnerRefAttr innerRef =
              getInnerRefTo(moduleLike, portIdx, getNamespace);
          auto path = updatePortTarget(moduleLike, anno, portIdx, innerRef);
          if (!path)
            return false;

          outputAnnotations.push_back({anno, *path});
          return true;
        });

    // Visit component annotations.
    moduleLike.walk([&](Operation *component) {
      AnnotationSet::removeAnnotations(component, [&](Annotation anno) {
        if (!anno.isClass(traceAnnoClass))
          return false;

        hw::InnerRefAttr innerRef = getInnerRefTo(component, getNamespace);
        auto path = updateTarget(moduleLike, component, anno, innerRef);
        if (!path)
          return false;

        outputAnnotations.push_back({anno, *path});
        return true;
      });
    });

    return outputAnnotations;
  };

  // Function to append one vector after another.  This is used to merge results
  // from parallel executions of "onModule".
  auto appendVecs = [](auto &&a, auto &&b) {
    a.append(b.begin(), b.end());
    return std::forward<decltype(a)>(a);
  };

  // Process all the modules in parallel or serially, depending on the
  // multithreading context.
  SmallVector<FModuleLike, 0> mods(circuit.getOps<FModuleLike>());
  auto outputAnnotations = transformReduce(
      context, mods, SmallVector<std::pair<Annotation, AnnoPathValue>>{},
      appendVecs, onModule);

  // Do not generate an output Annotation file if no Annotations exist.
  if (outputAnnotations.empty())
    return markAllAnalysesPreserved();

  // Write out all the Trace Annotations to a JSON buffer.
  std::string jsonBuffer;
  llvm::raw_string_ostream jsonStream(jsonBuffer);
  llvm::json::OStream json(jsonStream, /*IndentSize=*/2);
  json.arrayBegin();
  for (auto &[anno, path] : outputAnnotations) {
    json.objectBegin();
    json.attribute("class", anno.getClass());
    SmallString<64> targetStr;
    buildTarget(path, targetStr);
    LLVM_DEBUG({
      llvm::dbgs()
          << "  - chiselTarget: "
          << anno.getDict().getAs<StringAttr>("chiselTarget").getValue() << "\n"
          << "    target:       " << targetStr << "\n"
          << "    translated:   " << path << "\n";
    });
    json.attribute("target", targetStr);
    json.attribute("chiselTarget",
                   anno.getMember<StringAttr>("chiselTarget").getValue());
    json.objectEnd();
  }
  json.arrayEnd();

  LLVM_DEBUG({
    llvm::dbgs() << "Symbols:\n";
    for (auto [id, symbol] : llvm::enumerate(symbols))
      llvm::errs() << "  - " << id << ": " << symbol << "\n";
  });

  // Write the JSON-encoded Trace Annotation to a file called
  // "$circuitName.anno.json".  (This is implemented via an SVVerbatimOp that is
  // inserted before the FIRRTL circuit.
  auto b = OpBuilder::atBlockBegin(circuit.getBodyBlock());
  auto verbatimOp = b.create<sv::VerbatimOp>(
      b.getUnknownLoc(), jsonBuffer, ValueRange{}, b.getArrayAttr(symbols));
  hw::OutputFileAttr fileAttr;
  if (this->outputAnnotationFilename.empty())
    fileAttr = hw::OutputFileAttr::getFromFilename(
        context, circuit.getName() + ".anno.json",
        /*excludeFromFilelist=*/true, false);
  else
    fileAttr = hw::OutputFileAttr::getFromFilename(
        context, outputAnnotationFilename,
        /*excludeFromFilelist=*/true, false);
  verbatimOp->setAttr("output_file", fileAttr);

  return markAllAnalysesPreserved();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createResolveTracesPass(StringRef outputAnnotationFilename) {
  auto pass = std::make_unique<ResolveTracesPass>();
  if (!outputAnnotationFilename.empty())
    pass->outputAnnotationFilename = outputAnnotationFilename.str();
  return pass;
}
