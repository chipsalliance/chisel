//===- EmitOMIR.cpp ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the EmitOMIR pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotationHelper.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include <functional>

#define DEBUG_TYPE "omir"

using namespace circt;
using namespace firrtl;
using mlir::LocationAttr;
using mlir::UnitAttr;

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

namespace {
/// Information concerning a tracker in the IR.
struct Tracker {
  /// The unique ID of this tracker.
  IntegerAttr id;
  /// The operation onto which this tracker was annotated.
  Operation *op;
  /// If this tracker is non-local, this is the corresponding anchor.
  hw::HierPathOp nla;
  /// If this is a port, then set the portIdx, else initialized to -1.
  int portNo = -1;
  /// If this is a field, the ID will be greater than 0, else it will be 0.
  unsigned fieldID;

  // Returns true if the tracker has a non-zero field ID.
  bool hasFieldID() { return fieldID > 0; }
};

class EmitOMIRPass : public EmitOMIRBase<EmitOMIRPass> {
public:
  using EmitOMIRBase::outputFilename;

private:
  void runOnOperation() override;
  void makeTrackerAbsolute(Tracker &tracker);

  void emitSourceInfo(Location input, SmallString<64> &into);
  void emitOMNode(Attribute node, llvm::json::OStream &jsonStream);
  void emitOMField(StringAttr fieldName, DictionaryAttr field,
                   llvm::json::OStream &jsonStream);
  void emitOptionalRTLPorts(DictionaryAttr node,
                            llvm::json::OStream &jsonStream);
  void emitValue(Attribute node, llvm::json::OStream &jsonStream,
                 bool dutInstance);
  void emitTrackedTarget(DictionaryAttr node, llvm::json::OStream &jsonStream,
                         bool dutInstance);

  SmallString<8> addSymbolImpl(Attribute symbol) {
    unsigned id;
    auto it = symbolIndices.find(symbol);
    if (it != symbolIndices.end()) {
      id = it->second;
    } else {
      id = symbols.size();
      symbols.push_back(symbol);
      symbolIndices.insert({symbol, id});
    }
    SmallString<8> str;
    ("{{" + Twine(id) + "}}").toVector(str);
    return str;
  }
  SmallString<8> addSymbol(hw::InnerRefAttr symbol) {
    return addSymbolImpl(symbol);
  }
  SmallString<8> addSymbol(FlatSymbolRefAttr symbol) {
    return addSymbolImpl(symbol);
  }
  SmallString<8> addSymbol(StringAttr symbolName) {
    return addSymbol(FlatSymbolRefAttr::get(symbolName));
  }
  SmallString<8> addSymbol(Operation *op) {
    return addSymbol(SymbolTable::getSymbolName(op));
  }

  /// Obtain an inner reference to an operation, possibly adding an `inner_sym`
  /// to that operation.
  hw::InnerRefAttr getInnerRefTo(Operation *op);
  /// Obtain an inner reference to a module port, possibly adding an `inner_sym`
  /// to that port.
  hw::InnerRefAttr getInnerRefTo(FModuleLike module, size_t portIdx);

  // Obtain the result type of an Operation.
  FIRRTLType getTypeOf(Operation *op);
  // Obtain the type of a module port.
  FIRRTLType getTypeOf(FModuleLike mod, size_t portIdx);

  // Constructs a reference to a field from a FIRRTLType with a fieldID.
  void addFieldID(FIRRTLType type, unsigned fieldID,
                  SmallVectorImpl<char> &result);

  /// Get the cached namespace for a module.
  ModuleNamespace &getModuleNamespace(FModuleLike module) {
    auto it = moduleNamespaces.find(module);
    if (it != moduleNamespaces.end())
      return it->second;
    return moduleNamespaces.insert({module, ModuleNamespace(module)})
        .first->second;
  }

  /// Whether any errors have occurred in the current `runOnOperation`.
  bool anyFailures;
  CircuitNamespace *circuitNamespace;
  InstanceGraph *instanceGraph;
  InstancePathCache *instancePaths;
  /// OMIR target trackers gathered in the current operation, by tracker ID.
  DenseMap<Attribute, Tracker> trackers;
  /// The list of symbols to be interpolated in the verbatim JSON. This gets
  /// populated as the JSON is constructed and module and instance names are
  /// collected.
  SmallVector<Attribute> symbols;
  SmallDenseMap<Attribute, unsigned> symbolIndices;
  /// Temporary `firrtl.hierpath` operations to be deleted at the end of the
  /// pass. Vector elements are unique.
  SmallVector<hw::HierPathOp> removeTempNLAs;
  DenseMap<Operation *, ModuleNamespace> moduleNamespaces;
  /// Lookup table of instances by name and parent module.
  DenseMap<hw::InnerRefAttr, InstanceOp> instancesByName;
  /// Record to remove any temporary symbols added to instances.
  DenseSet<Operation *> tempSymInstances;
  /// The Design Under Test module.
  StringAttr dutModuleName;

  /// Cached NLA table analysis.
  NLATable *nlaTable;
};
} // namespace

/// Check if an `OMNode` is an `OMSRAM` and requires special treatment of its
/// instance path field. This returns the ID of the tracker stored in the
/// `instancePath` or `finalPath` field if the node has an array field `omType`
/// that contains a `OMString:OMSRAM` entry.
static IntegerAttr isOMSRAM(Attribute &node) {
  auto dict = dyn_cast<DictionaryAttr>(node);
  if (!dict)
    return {};
  auto idAttr = dict.getAs<StringAttr>("id");
  if (!idAttr)
    return {};
  IntegerAttr id;
  if (auto infoAttr = dict.getAs<DictionaryAttr>("fields")) {
    auto finalPath = infoAttr.getAs<DictionaryAttr>("finalPath");
    // The following is used prior to an upstream bump in Chisel.
    if (!finalPath)
      finalPath = infoAttr.getAs<DictionaryAttr>("instancePath");
    if (finalPath)
      if (auto v = finalPath.getAs<DictionaryAttr>("value"))
        if (v.getAs<UnitAttr>("omir.tracker"))
          id = v.getAs<IntegerAttr>("id");
    if (auto omTy = infoAttr.getAs<DictionaryAttr>("omType"))
      if (auto valueArr = omTy.getAs<ArrayAttr>("value"))
        for (auto attr : valueArr)
          if (auto str = dyn_cast<StringAttr>(attr))
            if (str.getValue().equals("OMString:OMSRAM"))
              return id;
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Code related to handling OMIR annotations
//===----------------------------------------------------------------------===//

/// Recursively walk Object Model IR and convert FIRRTL targets to identifiers
/// while scattering trackers into the newAnnotations argument.
///
/// Object Model IR consists of a type hierarchy built around recursive arrays
/// and dictionaries whose leaves are "string-encoded types".  This is an Object
/// Model-specific construct that puts type information alongside a value.
/// Concretely, these look like:
///
///     'OM' type ':' value
///
/// This function is only concerned with unpacking types whose values are FIRRTL
/// targets.  This is because these need to be kept up-to-date with
/// modifications made to the circuit whereas other types are just passing
/// through CIRCT.
///
/// At a later time this understanding may be expanded or Object Model IR may
/// become its own Dialect.  At this time, this function is trying to do as
/// minimal work as possible to just validate that the OMIR looks okay without
/// doing lots of unnecessary unpacking/repacking of string-encoded types.
static std::optional<Attribute> scatterOMIR(Attribute original,
                                            ApplyState &state) {
  auto *ctx = original.getContext();

  // Convert a string-encoded type to a dictionary that includes the type
  // information and an identifier derived from the current annotationID.  Then
  // increment the annotationID.  Return the constructed dictionary.
  auto addID = [&](StringRef tpe, StringRef path,
                   IntegerAttr id) -> DictionaryAttr {
    NamedAttrList fields;
    fields.append("id", id);
    fields.append("omir.tracker", UnitAttr::get(ctx));
    fields.append("path", StringAttr::get(ctx, path));
    fields.append("type", StringAttr::get(ctx, tpe));
    return DictionaryAttr::getWithSorted(ctx, fields);
  };

  return TypeSwitch<Attribute, std::optional<Attribute>>(original)
      // Most strings in the Object Model are actually string-encoded types.
      // These are types which look like: "<type>:<value>".  This code will
      // examine all strings, parse them into type and value, and then either
      // store them in their unpacked state (and possibly scatter trackers into
      // the circuit), store them in their packed state (because CIRCT is not
      // expected to care about them right now), or error if we see them
      // (because they should not exist and are expected to serialize to a
      // different format).
      .Case<StringAttr>([&](StringAttr str) -> std::optional<Attribute> {
        // Unpack the string into type and value.
        StringRef tpe, value;
        std::tie(tpe, value) = str.getValue().split(":");

        // These are string-encoded types that are targets in the circuit.
        // These require annotations to be scattered for them.  Replace their
        // target with an ID and scatter a tracker.
        if (tpe == "OMReferenceTarget" || tpe == "OMMemberReferenceTarget" ||
            tpe == "OMMemberInstanceTarget" || tpe == "OMInstanceTarget" ||
            tpe == "OMDontTouchedReferenceTarget") {
          auto idAttr = state.newID();
          NamedAttrList tracker;
          tracker.append("class", StringAttr::get(ctx, omirTrackerAnnoClass));
          tracker.append("id", idAttr);
          tracker.append("target", StringAttr::get(ctx, value));
          tracker.append("type", StringAttr::get(ctx, tpe));

          state.addToWorklistFn(DictionaryAttr::get(ctx, tracker));

          return addID(tpe, value, idAttr);
        }

        // The following are types that may exist, but we do not unbox them.  At
        // a later time, we may want to change this behavior and unbox these if
        // we wind up building out an Object Model dialect:
        if (isOMIRStringEncodedPassthrough(tpe))
          return str;

        // The following types are not expected to exist because they have
        // serializations to JSON types or are removed during serialization.
        // Hence, any of the following types are NOT expected to exist and we
        // error if we see them.  These are explicitly specified as opposed to
        // being handled in the "unknown" catch-all case below because we want
        // to provide a good error message that a user may be doing something
        // very weird.
        if (tpe == "OMMap" || tpe == "OMArray" || tpe == "OMBoolean" ||
            tpe == "OMInt" || tpe == "OMDouble" || tpe == "OMFrozenTarget") {
          auto diag =
              mlir::emitError(state.circuit.getLoc())
              << "found known string-encoded OMIR type \"" << tpe
              << "\", but this type should not be seen as it has a defined "
                 "serialization format that does NOT use a string-encoded type";
          diag.attachNote()
              << "the problematic OMIR is reproduced here: " << original;
          return std::nullopt;
        }

        // This is a catch-all for any unknown types.
        auto diag = mlir::emitError(state.circuit.getLoc())
                    << "found unknown string-encoded OMIR type \"" << tpe
                    << "\" (Did you misspell it?  Is CIRCT missing an Object "
                       "Model OMIR type?)";
        diag.attachNote() << "the problematic OMIR is reproduced here: "
                          << original;
        return std::nullopt;
      })
      // For an array, just recurse into each element and rewrite the array with
      // the results.
      .Case<ArrayAttr>([&](ArrayAttr arr) -> std::optional<Attribute> {
        SmallVector<Attribute> newArr;
        for (auto element : arr) {
          auto newElement = scatterOMIR(element, state);
          if (!newElement)
            return std::nullopt;
          newArr.push_back(*newElement);
        }
        return ArrayAttr::get(ctx, newArr);
      })
      // For a dictionary, recurse into each value and rewrite the key/value
      // pairs.
      .Case<DictionaryAttr>(
          [&](DictionaryAttr dict) -> std::optional<Attribute> {
            NamedAttrList newAttrs;
            for (auto pairs : dict) {
              auto maybeValue = scatterOMIR(pairs.getValue(), state);
              if (!maybeValue)
                return std::nullopt;
              newAttrs.append(pairs.getName(), *maybeValue);
            }
            return DictionaryAttr::get(ctx, newAttrs);
          })
      // These attributes are all expected.  They are OMIR types, but do not
      // have string-encodings (hence why these should error if we see them as
      // strings).
      .Case</* OMBoolean */ BoolAttr, /* OMDouble */ FloatAttr,
            /* OMInt */ IntegerAttr>(
          [](auto passThrough) { return passThrough; })
      // Error if we see anything else.
      .Default([&](auto) -> std::optional<Attribute> {
        auto diag = mlir::emitError(state.circuit.getLoc())
                    << "found unexpected MLIR attribute \"" << original
                    << "\" while trying to scatter OMIR";
        return std::nullopt;
      });
}

/// Convert an Object Model Field into an optional pair of a string key and a
/// dictionary attribute.  Expand internal source locator strings to location
/// attributes.  Scatter any FIRRTL targets into the circuit. If this is an
/// illegal Object Model Field return None.
///
/// Each Object Model Field consists of three mandatory members with
/// the following names and types:
///
///   - "info": Source Locator String
///   - "name": String
///   - "value": Object Model IR
///
/// The key is the "name" and the dictionary consists of the "info" and "value"
/// members.  Each value is recursively traversed to scatter any FIRRTL targets
/// that may be used inside it.
///
/// This conversion from an object (dictionary) to key--value pair is safe
/// because each Object Model Field in an Object Model Node must have a unique
/// "name".  Anything else is illegal Object Model.
static std::optional<std::pair<StringRef, DictionaryAttr>>
scatterOMField(Attribute original, const Attribute root, unsigned index,
               ApplyState &state) {
  // The input attribute must be a dictionary.
  DictionaryAttr dict = dyn_cast<DictionaryAttr>(original);
  if (!dict) {
    llvm::errs() << "OMField is not a dictionary, but should be: " << original
                 << "\n";
    return std::nullopt;
  }

  auto loc = state.circuit.getLoc();
  auto *ctx = state.circuit.getContext();

  // Generate an arbitrary identifier to use for caching when using
  // `maybeStringToLocation`.
  StringAttr locatorFilenameCache = StringAttr::get(ctx, ".");
  FileLineColLoc fileLineColLocCache;

  // Convert location from a string to a location attribute.
  auto infoAttr = tryGetAs<StringAttr>(dict, root, "info", loc, omirAnnoClass);
  if (!infoAttr)
    return std::nullopt;
  auto maybeLoc =
      maybeStringToLocation(infoAttr.getValue(), false, locatorFilenameCache,
                            fileLineColLocCache, ctx);
  mlir::LocationAttr infoLoc;
  if (maybeLoc.first)
    infoLoc = *maybeLoc.second;
  else
    infoLoc = UnknownLoc::get(ctx);

  // Extract the name attribute.
  auto nameAttr = tryGetAs<StringAttr>(dict, root, "name", loc, omirAnnoClass);
  if (!nameAttr)
    return std::nullopt;

  // The value attribute is unstructured and just copied over.
  auto valueAttr = tryGetAs<Attribute>(dict, root, "value", loc, omirAnnoClass);
  if (!valueAttr)
    return std::nullopt;
  auto newValue = scatterOMIR(valueAttr, state);
  if (!newValue)
    return std::nullopt;

  NamedAttrList values;
  // We add the index if one was provided.  This can be used later to
  // reconstruct the order of the original array.
  values.append("index", IntegerAttr::get(IntegerType::get(ctx, 64), index));
  values.append("info", infoLoc);
  values.append("value", *newValue);

  return {{nameAttr.getValue(), DictionaryAttr::getWithSorted(ctx, values)}};
}

/// Convert an Object Model Node to an optional dictionary, convert source
/// locator strings to location attributes, and scatter FIRRTL targets into the
/// circuit.  If this is an illegal Object Model Node, then return None.
///
/// An Object Model Node is expected to look like:
///
///   - "info": Source Locator String
///   - "id": String-encoded integer ('OMID' ':' Integer)
///   - "fields": Array<Object>
///
/// The "fields" member may be absent.  If so, then construct an empty array.
static std::optional<DictionaryAttr>
scatterOMNode(Attribute original, const Attribute root, ApplyState &state) {

  auto loc = state.circuit.getLoc();

  /// The input attribute must be a dictionary.
  DictionaryAttr dict = dyn_cast<DictionaryAttr>(original);
  if (!dict) {
    llvm::errs() << "OMNode is not a dictionary, but should be: " << original
                 << "\n";
    return std::nullopt;
  }

  NamedAttrList omnode;
  auto *ctx = state.circuit.getContext();

  // Generate an arbitrary identifier to use for caching when using
  // `maybeStringToLocation`.
  StringAttr locatorFilenameCache = StringAttr::get(ctx, ".");
  FileLineColLoc fileLineColLocCache;

  // Convert the location from a string to a location attribute.
  auto infoAttr = tryGetAs<StringAttr>(dict, root, "info", loc, omirAnnoClass);
  if (!infoAttr)
    return std::nullopt;
  auto maybeLoc =
      maybeStringToLocation(infoAttr.getValue(), false, locatorFilenameCache,
                            fileLineColLocCache, ctx);
  mlir::LocationAttr infoLoc;
  if (maybeLoc.first)
    infoLoc = *maybeLoc.second;
  else
    infoLoc = UnknownLoc::get(ctx);

  // Extract the OMID.  Don't parse this, just leave it as a string.
  auto idAttr = tryGetAs<StringAttr>(dict, root, "id", loc, omirAnnoClass);
  if (!idAttr)
    return std::nullopt;

  // Convert the fields from an ArrayAttr to a DictionaryAttr keyed by their
  // "name".  If no fields member exists, then just create an empty dictionary.
  // Note that this is safe to construct because all fields must have unique
  // "name" members relative to each other.
  auto maybeFields = dict.getAs<ArrayAttr>("fields");
  DictionaryAttr fields;
  if (!maybeFields)
    fields = DictionaryAttr::get(ctx);
  else {
    auto fieldAttr = maybeFields.getValue();
    NamedAttrList fieldAttrs;
    for (size_t i = 0, e = fieldAttr.size(); i != e; ++i) {
      auto field = fieldAttr[i];
      if (auto newField = scatterOMField(field, root, i, state)) {
        fieldAttrs.append(newField->first, newField->second);
        continue;
      }
      return std::nullopt;
    }
    fields = DictionaryAttr::get(ctx, fieldAttrs);
  }

  omnode.append("fields", fields);
  omnode.append("id", idAttr);
  omnode.append("info", infoLoc);

  return DictionaryAttr::getWithSorted(ctx, omnode);
}

/// Main entry point to handle scattering of an OMIRAnnotation.  Return the
/// modified optional attribute on success and None on failure.  Any scattered
/// annotations will be added to the reference argument `newAnnotations`.
LogicalResult circt::firrtl::applyOMIR(const AnnoPathValue &target,
                                       DictionaryAttr anno, ApplyState &state) {

  auto loc = state.circuit.getLoc();

  auto nodes = tryGetAs<ArrayAttr>(anno, anno, "nodes", loc, omirAnnoClass);
  if (!nodes)
    return failure();

  SmallVector<Attribute> newNodes;
  for (auto node : nodes) {
    auto newNode = scatterOMNode(node, anno, state);
    if (!newNode)
      return failure();
    newNodes.push_back(*newNode);
  }

  auto *ctx = state.circuit.getContext();

  NamedAttrList newAnnotation;
  newAnnotation.append("class", StringAttr::get(ctx, omirAnnoClass));
  newAnnotation.append("nodes", ArrayAttr::get(ctx, newNodes));

  AnnotationSet annotations(state.circuit);
  annotations.addAnnotations(DictionaryAttr::get(ctx, newAnnotation));
  annotations.applyToOperation(state.circuit);

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

void EmitOMIRPass::runOnOperation() {
  MLIRContext *context = &getContext();
  anyFailures = false;
  circuitNamespace = nullptr;
  instanceGraph = nullptr;
  instancePaths = nullptr;
  trackers.clear();
  symbols.clear();
  symbolIndices.clear();
  removeTempNLAs.clear();
  moduleNamespaces.clear();
  instancesByName.clear();
  CircuitOp circuitOp = getOperation();

  // Gather the relevant annotations from the circuit. On the one hand these are
  // all the actual `OMIRAnnotation`s that need processing and emission, as well
  // as an optional `OMIRFileAnnotation` that overrides the default OMIR output
  // file. Also while we're at it, keep track of all OMIR nodes that qualify as
  // an SRAM and that require their trackers to be turned into NLAs starting at
  // the root of the hierarchy.
  SmallVector<ArrayRef<Attribute>> annoNodes;
  DenseSet<Attribute> sramIDs;
  std::optional<StringRef> outputFilename;

  AnnotationSet::removeAnnotations(circuitOp, [&](Annotation anno) {
    if (anno.isClass(omirFileAnnoClass)) {
      auto pathAttr = anno.getMember<StringAttr>("filename");
      if (!pathAttr) {
        circuitOp.emitError(omirFileAnnoClass)
            << " annotation missing `filename` string attribute";
        anyFailures = true;
        return true;
      }
      LLVM_DEBUG(llvm::dbgs() << "- OMIR path: " << pathAttr << "\n");
      outputFilename = pathAttr.getValue();
      return true;
    }
    if (anno.isClass(omirAnnoClass)) {
      auto nodesAttr = anno.getMember<ArrayAttr>("nodes");
      if (!nodesAttr) {
        circuitOp.emitError(omirAnnoClass)
            << " annotation missing `nodes` array attribute";
        anyFailures = true;
        return true;
      }
      LLVM_DEBUG(llvm::dbgs() << "- OMIR: " << nodesAttr << "\n");
      annoNodes.push_back(nodesAttr.getValue());
      for (auto node : nodesAttr) {
        if (auto id = isOMSRAM(node)) {
          LLVM_DEBUG(llvm::dbgs() << "  - is SRAM with tracker " << id << "\n");
          sramIDs.insert(id);
        }
      }
      return true;
    }
    return false;
  });
  if (anyFailures)
    return signalPassFailure();

  // If an OMIR output filename has been specified as a pass parameter, override
  // whatever the annotations have configured. If neither are specified we just
  // bail.
  if (!this->outputFilename.empty())
    outputFilename = this->outputFilename;
  if (!outputFilename) {
    LLVM_DEBUG(llvm::dbgs() << "Not emitting OMIR because no annotation or "
                               "pass parameter specified an output file\n");
    markAllAnalysesPreserved();
    return;
  }
  // Establish some of the analyses we need throughout the pass.
  CircuitNamespace currentCircuitNamespace(circuitOp);
  InstanceGraph &currentInstanceGraph = getAnalysis<InstanceGraph>();
  nlaTable = &getAnalysis<NLATable>();
  InstancePathCache currentInstancePaths(currentInstanceGraph);
  circuitNamespace = &currentCircuitNamespace;
  instanceGraph = &currentInstanceGraph;
  instancePaths = &currentInstancePaths;
  dutModuleName = {};

  // Traverse the IR and collect all tracker annotations that were previously
  // scattered into the circuit.
  circuitOp.walk([&](Operation *op) {
    if (auto instOp = dyn_cast<InstanceOp>(op)) {
      // This instance does not have a symbol, but we are adding one. Remove it
      // after the pass.
      if (!op->getAttr(hw::InnerSymbolTable::getInnerSymbolAttrName()))
        tempSymInstances.insert(instOp);

      instancesByName.insert({getInnerRefTo(op), instOp});
    }
    auto setTracker = [&](int portNo, Annotation anno) {
      if (!anno.isClass(omirTrackerAnnoClass))
        return false;
      Tracker tracker;
      tracker.op = op;
      tracker.id = anno.getMember<IntegerAttr>("id");
      tracker.portNo = portNo;
      tracker.fieldID = anno.getFieldID();
      if (!tracker.id) {
        op->emitError(omirTrackerAnnoClass)
            << " annotation missing `id` integer attribute";
        anyFailures = true;
        return true;
      }
      if (auto nlaSym = anno.getMember<FlatSymbolRefAttr>("circt.nonlocal")) {
        auto tmp = nlaTable->getNLA(nlaSym.getAttr());
        if (!tmp) {
          op->emitError("missing annotation ") << nlaSym.getValue();
          anyFailures = true;
          return true;
        }
        tracker.nla = cast<hw::HierPathOp>(tmp);
      }
      if (sramIDs.erase(tracker.id))
        makeTrackerAbsolute(tracker);
      if (auto [it, inserted] = trackers.try_emplace(tracker.id, tracker);
          !inserted) {
        auto diag = op->emitError(omirTrackerAnnoClass)
                    << " annotation with same ID already found, must resolve "
                       "to single target";
        diag.attachNote(it->second.op->getLoc())
            << "tracker with same ID already found here";
        anyFailures = true;
        return true;
      }
      return true;
    };
    AnnotationSet::removePortAnnotations(op, setTracker);
    AnnotationSet::removeAnnotations(
        op, std::bind(setTracker, -1, std::placeholders::_1));
    if (auto modOp = dyn_cast<FModuleOp>(op)) {
      AnnotationSet annos(modOp.getAnnotations());
      if (!annos.hasAnnotation(dutAnnoClass))
        return;
      dutModuleName = modOp.getNameAttr();
    }
  });

  // Build the output JSON.
  std::string jsonBuffer;
  llvm::raw_string_ostream jsonOs(jsonBuffer);
  llvm::json::OStream json(jsonOs, 2);
  json.array([&] {
    for (auto nodes : annoNodes) {
      for (auto node : nodes) {
        emitOMNode(node, json);
        if (anyFailures)
          return;
      }
    }
  });
  if (anyFailures)
    return signalPassFailure();

  // Drop temporary (and sometimes invalid) NLA's created during the pass:
  for (auto nla : removeTempNLAs) {
    LLVM_DEBUG(llvm::dbgs() << "Removing '" << nla << "'\n");
    nlaTable->erase(nla);
    nla.erase();
  }
  removeTempNLAs.clear();

  // Remove the temp symbol from instances.
  for (auto *op : tempSymInstances)
    cast<InstanceOp>(op)->removeAttr("inner_sym");
  tempSymInstances.clear();

  // Emit the OMIR JSON as a verbatim op.
  auto builder = circuitOp.getBodyBuilder();
  auto verbatimOp =
      builder.create<sv::VerbatimOp>(builder.getUnknownLoc(), jsonBuffer);
  auto fileAttr = hw::OutputFileAttr::getFromFilename(
      context, *outputFilename, /*excludeFromFilelist=*/true, false);
  verbatimOp->setAttr("output_file", fileAttr);
  verbatimOp.setSymbolsAttr(ArrayAttr::get(context, symbols));

  markAnalysesPreserved<NLATable>();
}

/// Make a tracker absolute by adding an NLA to it which starts at the root
/// module of the circuit. Generates an error if any module along the path is
/// instantiated multiple times.
void EmitOMIRPass::makeTrackerAbsolute(Tracker &tracker) {
  auto builder = OpBuilder::atBlockBegin(getOperation().getBodyBlock());

  // Pick a name for the NLA that doesn't collide with anything.
  StringAttr opName;
  if (auto module = dyn_cast<FModuleLike>(tracker.op))
    opName = module.getModuleNameAttr();
  else
    opName = tracker.op->getAttrOfType<StringAttr>("name");
  auto nlaName = circuitNamespace->newName("omir_nla_" + opName.getValue());

  // Get all the paths instantiating this module. If there is an NLA already
  // attached to this tracker, we use it as a base to disambiguate the path to
  // the memory.
  hw::HWModuleLike mod;
  if (tracker.nla)
    mod = instanceGraph->lookup(tracker.nla.root())->getModule();
  else
    mod = tracker.op->getParentOfType<FModuleOp>();

  // Get all the paths instantiating this module.
  auto paths = instancePaths->getAbsolutePaths(mod);
  if (paths.empty()) {
    tracker.op->emitError("OMIR node targets uninstantiated component `")
        << opName.getValue() << "`";
    anyFailures = true;
    return;
  }
  if (paths.size() > 1) {
    auto diag = tracker.op->emitError("OMIR node targets ambiguous component `")
                << opName.getValue() << "`";
    diag.attachNote(tracker.op->getLoc())
        << "may refer to the following paths:";
    for (auto path : paths)
      formatInstancePath(diag.attachNote(tracker.op->getLoc()) << "- ", path);
    anyFailures = true;
    return;
  }

  // Assemble the module and name path for the NLA. Also attach an NLA reference
  // annotation to each instance participating in the path.
  SmallVector<Attribute> namepath;
  auto addToPath = [&](Operation *op) {
    namepath.push_back(getInnerRefTo(op));
  };
  // Add the path up to where the NLA starts.
  for (auto inst : paths[0])
    addToPath(inst);
  // Add the path from the NLA to the op.
  if (tracker.nla) {
    auto path = tracker.nla.getNamepath().getValue();
    for (auto attr : path.drop_back()) {
      auto ref = attr.cast<hw::InnerRefAttr>();
      // Find the instance referenced by the NLA.
      auto *node = instanceGraph->lookup(ref.getModule());
      auto it = llvm::find_if(*node, [&](hw::InstanceRecord *record) {
        return getInnerSymName(cast<InstanceOp>(*record->getInstance())) ==
               ref.getName();
      });
      assert(it != node->end() &&
             "Instance referenced by NLA does not exist in module");
      addToPath((*it)->getInstance());
    }
  }

  // TODO: Don't create NLA if namepath is empty
  // (care needed to ensure this will be handled correctly elsewhere)

  // Add the op itself.
  if (auto module = dyn_cast<FModuleLike>(tracker.op))
    namepath.push_back(FlatSymbolRefAttr::get(module.getModuleNameAttr()));
  else
    namepath.push_back(getInnerRefTo(tracker.op));

  // Add the NLA to the tracker and mark it to be deleted later.
  tracker.nla = builder.create<hw::HierPathOp>(builder.getUnknownLoc(),
                                               builder.getStringAttr(nlaName),
                                               builder.getArrayAttr(namepath));
  nlaTable->addNLA(tracker.nla);

  removeTempNLAs.push_back(tracker.nla);
}

/// Emit a source locator into a string, for inclusion in the `info` field of
/// `OMNode` and `OMField`.
void EmitOMIRPass::emitSourceInfo(Location input, SmallString<64> &into) {
  into.clear();
  input->walk([&](Location loc) {
    if (FileLineColLoc fileLoc = dyn_cast<FileLineColLoc>(loc)) {
      into.append(into.empty() ? "@[" : " ");
      (Twine(fileLoc.getFilename()) + " " + Twine(fileLoc.getLine()) + ":" +
       Twine(fileLoc.getColumn()))
          .toVector(into);
    }
    return WalkResult::advance();
  });
  if (!into.empty())
    into.append("]");
  else
    into.append("UnlocatableSourceInfo");
}

/// Emit an entire `OMNode` as JSON.
void EmitOMIRPass::emitOMNode(Attribute node, llvm::json::OStream &jsonStream) {
  auto dict = dyn_cast<DictionaryAttr>(node);
  if (!dict) {
    getOperation()
            .emitError("OMNode must be a dictionary")
            .attachNote(getOperation().getLoc())
        << node;
    anyFailures = true;
    return;
  }

  // Extract the `info` field and serialize the location.
  SmallString<64> info;
  if (auto infoAttr = dict.getAs<LocationAttr>("info"))
    emitSourceInfo(infoAttr, info);
  if (anyFailures)
    return;

  // Extract the `id` field.
  auto idAttr = dict.getAs<StringAttr>("id");
  if (!idAttr) {
    getOperation()
            .emitError("OMNode missing `id` string field")
            .attachNote(getOperation().getLoc())
        << dict;
    anyFailures = true;
    return;
  }

  // Extract and order the fields of this node.
  SmallVector<std::tuple<unsigned, StringAttr, DictionaryAttr>> orderedFields;
  auto fieldsDict = dict.getAs<DictionaryAttr>("fields");
  if (fieldsDict) {
    for (auto nameAndField : fieldsDict.getValue()) {
      auto fieldDict = dyn_cast<DictionaryAttr>(nameAndField.getValue());
      if (!fieldDict) {
        getOperation()
                .emitError("OMField must be a dictionary")
                .attachNote(getOperation().getLoc())
            << nameAndField.getValue();
        anyFailures = true;
        return;
      }

      unsigned index = 0;
      if (auto indexAttr = fieldDict.getAs<IntegerAttr>("index"))
        index = indexAttr.getValue().getLimitedValue();

      orderedFields.push_back({index, nameAndField.getName(), fieldDict});
    }
    llvm::sort(orderedFields,
               [](auto a, auto b) { return std::get<0>(a) < std::get<0>(b); });
  }

  jsonStream.object([&] {
    jsonStream.attribute("info", info);
    jsonStream.attribute("id", idAttr.getValue());
    jsonStream.attributeArray("fields", [&] {
      for (auto &orderedField : orderedFields) {
        emitOMField(std::get<1>(orderedField), std::get<2>(orderedField),
                    jsonStream);
        if (anyFailures)
          return;
      }
      if (auto node = fieldsDict.getAs<DictionaryAttr>("containingModule"))
        if (auto value = node.getAs<DictionaryAttr>("value"))
          emitOptionalRTLPorts(value, jsonStream);
    });
  });
}

/// Emit a single `OMField` as JSON. This expects the field's name to be
/// provided from the outside, for example as the field name that this attribute
/// has in the surrounding dictionary.
void EmitOMIRPass::emitOMField(StringAttr fieldName, DictionaryAttr field,
                               llvm::json::OStream &jsonStream) {
  // Extract the `info` field and serialize the location.
  auto infoAttr = field.getAs<LocationAttr>("info");
  SmallString<64> info;
  if (infoAttr)
    emitSourceInfo(infoAttr, info);
  if (anyFailures)
    return;

  jsonStream.object([&] {
    jsonStream.attribute("info", info);
    jsonStream.attribute("name", fieldName.strref());
    jsonStream.attributeBegin("value");
    emitValue(field.get("value"), jsonStream,
              fieldName.strref().equals("dutInstance"));
    jsonStream.attributeEnd();
  });
}

// If the given `node` refers to a valid tracker in the IR, gather the
// additional port metadata of the module it refers to. Then emit this port
// metadata as a `ports` array field for the surrounding `OMNode`.
void EmitOMIRPass::emitOptionalRTLPorts(DictionaryAttr node,
                                        llvm::json::OStream &jsonStream) {
  // First make sure we actually have a valid tracker. If not, just silently
  // abort and don't emit any port metadata.
  auto idAttr = node.getAs<IntegerAttr>("id");
  auto trackerIt = trackers.find(idAttr);
  if (!idAttr || !node.getAs<UnitAttr>("omir.tracker") ||
      trackerIt == trackers.end())
    return;
  auto tracker = trackerIt->second;

  // Lookup the module the tracker refers to. If it points at something *within*
  // a module, go dig up the surrounding module. This is roughly what
  // `Target.referringModule(...)` does on the Scala side.
  auto module = dyn_cast<FModuleLike>(tracker.op);
  if (!module)
    module = tracker.op->getParentOfType<FModuleLike>();
  if (!module) {
    LLVM_DEBUG(llvm::dbgs() << "Not emitting RTL ports since tracked operation "
                               "does not have a FModuleLike parent: "
                            << *tracker.op << "\n");
    return;
  }
  LLVM_DEBUG(llvm::dbgs() << "Emitting RTL ports for module `"
                          << module.getModuleName() << "`\n");

  // Emit the JSON.
  SmallString<64> buf;
  jsonStream.object([&] {
    buf.clear();
    emitSourceInfo(module.getLoc(), buf);
    jsonStream.attribute("info", buf);
    jsonStream.attribute("name", "ports");
    jsonStream.attributeArray("value", [&] {
      for (const auto &port : llvm::enumerate(module.getPorts())) {
        auto portType = type_dyn_cast<FIRRTLBaseType>(port.value().type);
        if (!portType || portType.getBitWidthOrSentinel() == 0)
          continue;
        jsonStream.object([&] {
          // Emit the `ref` field.
          buf.assign("OMDontTouchedReferenceTarget:~");
          if (module.getModuleNameAttr() == dutModuleName) {
            // If module is DUT, then root the target relative to the DUT.
            buf.append(module.getModuleName());
          } else {
            buf.append(getOperation().getName());
          }
          buf.push_back('|');
          buf.append(addSymbol(module));
          buf.push_back('>');
          buf.append(addSymbol(getInnerRefTo(module, port.index())));
          jsonStream.attribute("ref", buf);

          // Emit the `direction` field.
          buf.assign("OMString:");
          buf.append(port.value().isOutput() ? "Output" : "Input");
          jsonStream.attribute("direction", buf);

          // Emit the `width` field.
          buf.assign("OMBigInt:");
          Twine::utohexstr(portType.getBitWidthOrSentinel()).toVector(buf);
          jsonStream.attribute("width", buf);
        });
      }
    });
  });
}

void EmitOMIRPass::emitValue(Attribute node, llvm::json::OStream &jsonStream,
                             bool dutInstance) {
  // Handle the null case.
  if (!node || isa<UnitAttr>(node))
    return jsonStream.value(nullptr);

  // Handle the trivial cases where the OMIR serialization simply uses the
  // builtin JSON types.
  if (auto attr = dyn_cast<BoolAttr>(node))
    return jsonStream.value(attr.getValue()); // OMBoolean
  if (auto attr = dyn_cast<IntegerAttr>(node)) {
    // CAVEAT: We expect these integers to come from an OMIR file that is
    // initially read in from JSON, where they are i32 or i64, so this should
    // yield a valid value. However, a user could cook up an arbitrary precision
    // integer attr in MLIR input and then subtly break the JSON spec.
    SmallString<16> val;
    attr.getValue().toStringSigned(val);
    return jsonStream.rawValue(val); // OMInt
  }
  if (auto attr = dyn_cast<FloatAttr>(node)) {
    // CAVEAT: We expect these floats to come from an OMIR file that is
    // initially read in from JSON, where they are f32 or f64, so this should
    // yield a valid value. However, a user could cook up an arbitrary precision
    // float attr in MLIR input and then subtly break the JSON spec.
    SmallString<16> val;
    attr.getValue().toString(val);
    return jsonStream.rawValue(val); // OMDouble
  }

  // Handle aggregate types.
  if (auto attr = dyn_cast<ArrayAttr>(node)) {
    jsonStream.array([&] {
      for (auto element : attr.getValue()) {
        emitValue(element, jsonStream, dutInstance);
        if (anyFailures)
          return;
      }
    });
    return;
  }
  if (auto attr = dyn_cast<DictionaryAttr>(node)) {
    // Handle targets that have a corresponding tracker annotation in the IR.
    if (attr.getAs<UnitAttr>("omir.tracker"))
      return emitTrackedTarget(attr, jsonStream, dutInstance);

    // Handle regular dictionaries.
    jsonStream.object([&] {
      for (auto field : attr.getValue()) {
        jsonStream.attributeBegin(field.getName());
        emitValue(field.getValue(), jsonStream, dutInstance);
        jsonStream.attributeEnd();
        if (anyFailures)
          return;
      }
    });
    return;
  }

  // The remaining types are all simple string-encoded pass-through cases.
  if (auto attr = dyn_cast<StringAttr>(node)) {
    StringRef val = attr.getValue();
    if (isOMIRStringEncodedPassthrough(val.split(":").first))
      return jsonStream.value(val);
  }

  // If we get here, we don't know how to serialize the given MLIR attribute as
  // a OMIR value.
  jsonStream.value("<unsupported value>");
  getOperation().emitError("unsupported attribute for OMIR serialization: `")
      << node << "`";
  anyFailures = true;
}

void EmitOMIRPass::emitTrackedTarget(DictionaryAttr node,
                                     llvm::json::OStream &jsonStream,
                                     bool dutInstance) {
  // Extract the `id` field.
  auto idAttr = node.getAs<IntegerAttr>("id");
  if (!idAttr) {
    getOperation()
            .emitError("tracked OMIR target missing `id` string field")
            .attachNote(getOperation().getLoc())
        << node;
    anyFailures = true;
    return jsonStream.value("<error>");
  }

  // Extract the `type` field.
  auto typeAttr = node.getAs<StringAttr>("type");
  if (!typeAttr) {
    getOperation()
            .emitError("tracked OMIR target missing `type` string field")
            .attachNote(getOperation().getLoc())
        << node;
    anyFailures = true;
    return jsonStream.value("<error>");
  }
  StringRef type = typeAttr.getValue();

  // Find the tracker for this target, and handle the case where the tracker has
  // been deleted.
  auto trackerIt = trackers.find(idAttr);
  if (trackerIt == trackers.end()) {
    // Some of the target types indicate removal of the target through an
    // `OMDeleted` node.
    if (type == "OMReferenceTarget" || type == "OMMemberReferenceTarget" ||
        type == "OMMemberInstanceTarget")
      return jsonStream.value("OMDeleted:");

    // The remaining types produce an error upon removal of the target.
    auto diag = getOperation().emitError("tracked OMIR target of type `")
                << type << "` was deleted";
    diag.attachNote(getOperation().getLoc())
        << "`" << type << "` should never be deleted";
    if (auto path = node.getAs<StringAttr>("path"))
      diag.attachNote(getOperation().getLoc())
          << "original path: `" << path.getValue() << "`";
    anyFailures = true;
    return jsonStream.value("<error>");
  }
  auto tracker = trackerIt->second;

  // In case this is an `OMMemberTarget`, handle the case where the component
  // used to be a "reference target" (wire, register, memory, node) when the
  // OMIR was read in, but has been change to an "instance target" during the
  // execution of the compiler. This mainly occurs during mapping of
  // `firrtl.mem` operations to a corresponding `firrtl.instance`.
  if (type == "OMMemberReferenceTarget" && isa<InstanceOp, MemOp>(tracker.op))
    type = "OMMemberInstanceTarget";

  // Serialize the target circuit first.
  SmallString<64> target(type);
  target.append(":~");
  target.append(getOperation().getName());
  target.push_back('|');

  // Serialize the local or non-local module/instance hierarchy path.
  if (tracker.nla) {
    bool notFirst = false;
    hw::InnerRefAttr instName;
    for (auto nameRef : tracker.nla.getNamepath()) {
      StringAttr modName;
      if (auto innerRef = nameRef.dyn_cast<hw::InnerRefAttr>())
        modName = innerRef.getModule();
      else if (auto ref = dyn_cast<FlatSymbolRefAttr>(nameRef))
        modName = ref.getAttr();
      if (!dutInstance && modName == dutModuleName) {
        // Check if the DUT module occurs in the instance path.
        // Print the path relative to the DUT, if the nla is inside the DUT.
        // Keep the path for dutInstance relative to test harness. (SFC
        // implementation in TestHarnessOMPhase.scala)
        target = type;
        target.append(":~");
        target.append(dutModuleName);
        target.push_back('|');
        notFirst = false;
        instName = {};
      }

      Operation *module = nlaTable->getModule(modName);
      assert(module);
      if (notFirst)
        target.push_back('/');
      notFirst = true;
      if (instName) {
        target.append(addSymbol(instName));
        target.push_back(':');
      }
      target.append(addSymbol(module));

      if (auto innerRef = nameRef.dyn_cast<hw::InnerRefAttr>()) {
        // Find an instance with the given name in this module. Ensure it has a
        // symbol that we can refer to.
        auto instOp = instancesByName.lookup(innerRef);
        if (!instOp)
          continue;
        LLVM_DEBUG(llvm::dbgs() << "Marking NLA-participating instance "
                                << innerRef.getName() << " in module "
                                << modName << " as dont-touch\n");
        tempSymInstances.erase(instOp);
        instName = getInnerRefTo(instOp);
      }
    }
  } else {
    FModuleOp module = dyn_cast<FModuleOp>(tracker.op);
    if (!module)
      module = tracker.op->getParentOfType<FModuleOp>();
    assert(module);
    if (module.getNameAttr() == dutModuleName) {
      // If module is DUT, then root the target relative to the DUT.
      target = type;
      target.append(":~");
      target.append(dutModuleName);
      target.push_back('|');
    }
    target.append(addSymbol(module));
  }

  // Serialize any potential component *inside* the module that this target may
  // specifically refer to.
  hw::InnerRefAttr componentName;
  FIRRTLType componentType;
  if (isa<WireOp, RegOp, RegResetOp, InstanceOp, NodeOp, MemOp>(tracker.op)) {
    tempSymInstances.erase(tracker.op);
    componentName = getInnerRefTo(tracker.op);
    LLVM_DEBUG(llvm::dbgs() << "Marking OMIR-targeted " << componentName
                            << " as dont-touch\n");

    // If the target refers to a field, get the type of the component so we can
    // extract the field, or fail if we don't know how to get the type.
    if (tracker.hasFieldID()) {
      if (isa<WireOp, RegOp, RegResetOp, NodeOp>(tracker.op)) {
        componentType = getTypeOf(tracker.op);
      } else {
        tracker.op->emitError("does not support OMIR targeting fields");
        anyFailures = true;
        return jsonStream.value("<error>");
      }
    }
  } else if (auto mod = dyn_cast<FModuleLike>(tracker.op)) {
    if (tracker.portNo >= 0) {
      componentName = getInnerRefTo(mod, tracker.portNo);

      // If the target refers to a field, get the type of the port.
      if (tracker.hasFieldID())
        componentType = getTypeOf(mod, tracker.portNo);
    }
  } else if (!isa<FModuleLike>(tracker.op)) {
    tracker.op->emitError("invalid target for `") << type << "` OMIR";
    anyFailures = true;
    return jsonStream.value("<error>");
  }
  if (componentName) {
    // Check if the targeted component is going to be emitted as an instance.
    // This is trivially the case for `InstanceOp`s, but also for `MemOp`s that
    // get converted to an instance during lowering to HW dialect and generator
    // callout.
    [&] {
      if (type == "OMMemberInstanceTarget") {
        if (auto instOp = dyn_cast<InstanceOp>(tracker.op)) {
          target.push_back('/');
          target.append(addSymbol(componentName));
          target.push_back(':');
          target.append(addSymbol(instOp.getModuleNameAttr()));
          return;
        }
        if (auto memOp = dyn_cast<MemOp>(tracker.op)) {
          target.push_back('/');
          target.append(addSymbol(componentName));
          target.push_back(':');
          target.append(memOp.getSummary().getFirMemoryName());
          return;
        }
      }
      target.push_back('>');
      target.append(addSymbol(componentName));
      addFieldID(componentType, tracker.fieldID, target);
    }();
  }

  jsonStream.value(target);
}

hw::InnerRefAttr EmitOMIRPass::getInnerRefTo(Operation *op) {
  return ::getInnerRefTo(op, [&](FModuleOp module) -> ModuleNamespace & {
    return getModuleNamespace(module);
  });
}

hw::InnerRefAttr EmitOMIRPass::getInnerRefTo(FModuleLike module,
                                             size_t portIdx) {
  return ::getInnerRefTo(module, portIdx,
                         [&](FModuleLike mod) -> ModuleNamespace & {
                           return getModuleNamespace(mod);
                         });
}

FIRRTLType EmitOMIRPass::getTypeOf(Operation *op) {
  if (auto fop = dyn_cast<Forceable>(op))
    return fop.getDataType();
  assert(op->getNumResults() == 1 &&
         isa<FIRRTLType>(op->getResult(0).getType()) &&
         "op must have a single FIRRTLType result");
  return type_cast<FIRRTLType>(op->getResult(0).getType());
}

FIRRTLType EmitOMIRPass::getTypeOf(FModuleLike mod, size_t portIdx) {
  Type portType = mod.getPortType(portIdx);
  assert(isa<FIRRTLType>(portType) && "port must have a FIRRTLType");
  return type_cast<FIRRTLType>(portType);
}

// Constructs a reference to a field of an aggregate FIRRTLType with a fieldID,
// and appends it to result. If fieldID is 0, meaning it does not reference a
// field of an aggregate FIRRTLType, this is a no-op.
void EmitOMIRPass::addFieldID(FIRRTLType type, unsigned fieldID,
                              SmallVectorImpl<char> &result) {
  while (fieldID)
    FIRRTLTypeSwitch<FIRRTLType>(type)
        .Case<FVectorType>([&](FVectorType vector) {
          size_t index = vector.getIndexForFieldID(fieldID);
          type = vector.getElementType();
          fieldID -= vector.getFieldID(index);
          result.push_back('[');
          Twine(index).toVector(result);
          result.push_back(']');
        })
        .Case<BundleType>([&](BundleType bundle) {
          size_t index = bundle.getIndexForFieldID(fieldID);
          StringRef name = bundle.getElement(index).name;
          type = bundle.getElementType(index);
          fieldID -= bundle.getFieldID(index);
          result.push_back('.');
          result.append(name.begin(), name.end());
        });
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass>
circt::firrtl::createEmitOMIRPass(StringRef outputFilename) {
  auto pass = std::make_unique<EmitOMIRPass>();
  if (!outputFilename.empty())
    pass->outputFilename = outputFilename.str();
  return pass;
}
