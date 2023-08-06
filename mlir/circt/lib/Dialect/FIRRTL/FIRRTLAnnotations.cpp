//===- FIRRTLAnnotations.cpp - Code for working with Annotations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements helpers for working with FIRRTL annotations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace firrtl;

static ArrayAttr getAnnotationsFrom(Operation *op) {
  if (auto annots = op->getAttrOfType<ArrayAttr>(getAnnotationAttrName()))
    return annots;
  return ArrayAttr::get(op->getContext(), {});
}

static ArrayAttr getAnnotationsFrom(ArrayRef<Annotation> annotations,
                                    MLIRContext *context) {
  if (annotations.empty())
    return ArrayAttr::get(context, {});
  SmallVector<Attribute> attrs;
  attrs.reserve(annotations.size());
  for (auto anno : annotations)
    attrs.push_back(anno.getAttr());
  return ArrayAttr::get(context, attrs);
}

/// Form an annotation set from an array of annotation attributes.
AnnotationSet::AnnotationSet(ArrayRef<Attribute> annotations,
                             MLIRContext *context)
    : annotations(ArrayAttr::get(context, annotations)) {}

/// Form an annotation set from an array of annotations.
AnnotationSet::AnnotationSet(ArrayRef<Annotation> annotations,
                             MLIRContext *context)
    : annotations(getAnnotationsFrom(annotations, context)) {}

/// Form an annotation set with a possibly-null ArrayAttr.
AnnotationSet::AnnotationSet(ArrayAttr annotations, MLIRContext *context)
    : AnnotationSet(annotations ? annotations : ArrayAttr::get(context, {})) {}

/// Get an annotation set for the specified operation.
AnnotationSet::AnnotationSet(Operation *op)
    : AnnotationSet(getAnnotationsFrom(op)) {}

static AnnotationSet forPort(Operation *op, size_t portNo) {
  auto ports = op->getAttrOfType<ArrayAttr>(getPortAnnotationAttrName());
  if (ports && !ports.empty())
    return AnnotationSet(cast<ArrayAttr>(ports[portNo]));
  return AnnotationSet(ArrayAttr::get(op->getContext(), {}));
}

AnnotationSet AnnotationSet::forPort(FModuleLike op, size_t portNo) {
  return ::forPort(op.getOperation(), portNo);
}

AnnotationSet AnnotationSet::forPort(MemOp op, size_t portNo) {
  return ::forPort(op.getOperation(), portNo);
}

/// Get an annotation set for the specified value.
AnnotationSet AnnotationSet::get(Value v) {
  if (auto op = v.getDefiningOp())
    return AnnotationSet(op);
  // If its not an Operation, then must be a block argument.
  auto arg = dyn_cast<BlockArgument>(v);
  auto module = cast<FModuleOp>(arg.getOwner()->getParentOp());
  return forPort(module, arg.getArgNumber());
}

/// Store the annotations in this set in an operation's `annotations` attribute,
/// overwriting any existing annotations.
bool AnnotationSet::applyToOperation(Operation *op) const {
  auto before = op->getAttrDictionary();
  op->setAttr(getAnnotationAttrName(), getArrayAttr());
  return op->getAttrDictionary() != before;
}

static bool applyToPort(AnnotationSet annos, Operation *op, size_t portCount,
                        size_t portNo) {
  assert(portNo < portCount && "port index out of range.");
  auto *context = op->getContext();
  auto before = op->getAttrOfType<ArrayAttr>(getPortAnnotationAttrName());
  SmallVector<Attribute> portAnnotations;
  if (!before || before.empty())
    portAnnotations.assign(portCount, ArrayAttr::get(context, {}));
  else
    portAnnotations.append(before.begin(), before.end());
  portAnnotations[portNo] = annos.getArrayAttr();
  auto after = ArrayAttr::get(context, portAnnotations);
  op->setAttr(getPortAnnotationAttrName(), after);
  return before != after;
}

bool AnnotationSet::applyToPort(FModuleLike op, size_t portNo) const {
  return ::applyToPort(*this, op.getOperation(), getNumPorts(op), portNo);
}

bool AnnotationSet::applyToPort(MemOp op, size_t portNo) const {
  return ::applyToPort(*this, op.getOperation(), op->getNumResults(), portNo);
}

static bool applyToAttrListImpl(const AnnotationSet &annoSet, StringRef key,
                                NamedAttrList &attrs) {
  if (annoSet.empty())
    return bool(attrs.erase(key));
  else {
    auto attr = annoSet.getArrayAttr();
    return attrs.set(key, attr) != attr;
  }
}

/// Store the annotations in this set in a `NamedAttrList` as an array attribute
/// with the name `annotations`.
bool AnnotationSet::applyToAttrList(NamedAttrList &attrs) const {
  return applyToAttrListImpl(*this, getAnnotationAttrName(), attrs);
}

/// Store the annotations in this set in a `NamedAttrList` as an array attribute
/// with the name `firrtl.annotations`.
bool AnnotationSet::applyToPortAttrList(NamedAttrList &attrs) const {
  return applyToAttrListImpl(*this, getDialectAnnotationAttrName(), attrs);
}

static DictionaryAttr applyToDictionaryAttrImpl(const AnnotationSet &annoSet,
                                                StringRef key,
                                                ArrayRef<NamedAttribute> attrs,
                                                bool sorted,
                                                DictionaryAttr originalDict) {
  // Find the location in the dictionary where the entry would go.
  ArrayRef<NamedAttribute>::iterator it;
  if (sorted) {
    it = llvm::lower_bound(attrs, key);
    if (it != attrs.end() && it->getName() != key)
      it = attrs.end();
  } else {
    it = llvm::find_if(
        attrs, [key](NamedAttribute attr) { return attr.getName() == key; });
  }

  // Fast path in case there are no annotations in the dictionary and we are not
  // supposed to add any.
  if (it == attrs.end() && annoSet.empty())
    return originalDict;

  // Fast path in case there already is an entry in the dictionary, it matches
  // the set, and, in the case we're supposed to remove empty sets, we're not
  // leaving an empty entry in the dictionary.
  if (it != attrs.end() && it->getValue() == annoSet.getArrayAttr() &&
      !annoSet.empty())
    return originalDict;

  // If we arrive here, we are supposed to assemble a new dictionary.
  SmallVector<NamedAttribute> newAttrs;
  newAttrs.reserve(attrs.size() + 1);
  newAttrs.append(attrs.begin(), it);
  if (!annoSet.empty())
    newAttrs.push_back(
        {StringAttr::get(annoSet.getContext(), key), annoSet.getArrayAttr()});
  if (it != attrs.end())
    newAttrs.append(it + 1, attrs.end());
  return sorted ? DictionaryAttr::getWithSorted(annoSet.getContext(), newAttrs)
                : DictionaryAttr::get(annoSet.getContext(), newAttrs);
}

/// Update the attribute dictionary of an operation to contain this annotation
/// set.
DictionaryAttr
AnnotationSet::applyToDictionaryAttr(DictionaryAttr attrs) const {
  return applyToDictionaryAttrImpl(*this, getAnnotationAttrName(),
                                   attrs.getValue(), true, attrs);
}

DictionaryAttr
AnnotationSet::applyToDictionaryAttr(ArrayRef<NamedAttribute> attrs) const {
  return applyToDictionaryAttrImpl(*this, getAnnotationAttrName(), attrs, false,
                                   {});
}

/// Update the attribute dictionary of a port to contain this annotation set.
DictionaryAttr
AnnotationSet::applyToPortDictionaryAttr(DictionaryAttr attrs) const {
  return applyToDictionaryAttrImpl(*this, getDialectAnnotationAttrName(),
                                   attrs.getValue(), true, attrs);
}

DictionaryAttr
AnnotationSet::applyToPortDictionaryAttr(ArrayRef<NamedAttribute> attrs) const {
  return applyToDictionaryAttrImpl(*this, getDialectAnnotationAttrName(), attrs,
                                   false, {});
}

Annotation AnnotationSet::getAnnotationImpl(StringAttr className) const {
  for (auto annotation : *this) {
    if (annotation.getClassAttr() == className)
      return annotation;
  }
  return {};
}

Annotation AnnotationSet::getAnnotationImpl(StringRef className) const {
  for (auto annotation : *this) {
    if (annotation.getClass() == className)
      return annotation;
  }
  return {};
}

bool AnnotationSet::hasAnnotationImpl(StringAttr className) const {
  return getAnnotationImpl(className) != Annotation();
}

bool AnnotationSet::hasAnnotationImpl(StringRef className) const {
  return getAnnotationImpl(className) != Annotation();
}

bool AnnotationSet::hasDontTouch() const {
  return hasAnnotation(dontTouchAnnoClass);
}

bool AnnotationSet::setDontTouch(bool dontTouch) {
  if (dontTouch)
    return addDontTouch();
  else
    return removeDontTouch();
}

bool AnnotationSet::addDontTouch() {
  if (hasDontTouch())
    return false;
  addAnnotations(DictionaryAttr::get(
      getContext(), {{StringAttr::get(getContext(), "class"),
                      StringAttr::get(getContext(), dontTouchAnnoClass)}}));
  return true;
}

bool AnnotationSet::removeDontTouch() {
  return removeAnnotation(dontTouchAnnoClass);
}

bool AnnotationSet::hasDontTouch(Operation *op) {
  return AnnotationSet(op).hasDontTouch();
}

bool AnnotationSet::setDontTouch(Operation *op, bool dontTouch) {
  if (dontTouch)
    return addDontTouch(op);
  else
    return removeDontTouch(op);
}

bool AnnotationSet::addDontTouch(Operation *op) {
  AnnotationSet annos(op);
  auto changed = annos.addDontTouch();
  if (changed)
    annos.applyToOperation(op);
  return changed;
}

bool AnnotationSet::removeDontTouch(Operation *op) {
  AnnotationSet annos(op);
  auto changed = annos.removeDontTouch();
  if (changed)
    annos.applyToOperation(op);
  return changed;
}

/// Add more annotations to this AttributeSet.
void AnnotationSet::addAnnotations(ArrayRef<Annotation> newAnnotations) {
  if (newAnnotations.empty())
    return;

  SmallVector<Attribute> annotationVec;
  annotationVec.reserve(annotations.size() + newAnnotations.size());
  annotationVec.append(annotations.begin(), annotations.end());
  for (auto anno : newAnnotations)
    annotationVec.push_back(anno.getDict());
  annotations = ArrayAttr::get(getContext(), annotationVec);
}

void AnnotationSet::addAnnotations(ArrayRef<Attribute> newAnnotations) {
  if (newAnnotations.empty())
    return;

  if (empty()) {
    annotations = ArrayAttr::get(getContext(), newAnnotations);
    return;
  }

  SmallVector<Attribute> annotationVec;
  annotationVec.reserve(annotations.size() + newAnnotations.size());
  annotationVec.append(annotations.begin(), annotations.end());
  annotationVec.append(newAnnotations.begin(), newAnnotations.end());
  annotations = ArrayAttr::get(getContext(), annotationVec);
}

void AnnotationSet::addAnnotations(ArrayAttr newAnnotations) {
  if (!newAnnotations)
    return;

  if (empty()) {
    annotations = newAnnotations;
    return;
  }

  SmallVector<Attribute> annotationVec;
  annotationVec.reserve(annotations.size() + newAnnotations.size());
  annotationVec.append(annotations.begin(), annotations.end());
  annotationVec.append(newAnnotations.begin(), newAnnotations.end());
  annotations = ArrayAttr::get(getContext(), annotationVec);
}

/// Remove an annotation from this annotation set. Returns true if any were
/// removed, false otherwise.
bool AnnotationSet::removeAnnotation(Annotation anno) {
  return removeAnnotations([&](Annotation other) { return other == anno; });
}

/// Remove an annotation from this annotation set. Returns true if any were
/// removed, false otherwise.
bool AnnotationSet::removeAnnotation(Attribute anno) {
  return removeAnnotations(
      [&](Annotation other) { return other.getDict() == anno; });
}

/// Remove an annotation from this annotation set. Returns true if any were
/// removed, false otherwise.
bool AnnotationSet::removeAnnotation(StringRef className) {
  return removeAnnotations(
      [&](Annotation other) { return other.getClass() == className; });
}

/// Remove all annotations from this annotation set for which `predicate`
/// returns true.
bool AnnotationSet::removeAnnotations(
    llvm::function_ref<bool(Annotation)> predicate) {
  // Fast path for empty sets.
  auto attr = getArrayAttr();
  if (!attr)
    return false;

  // Search for the first match.
  ArrayRef<Attribute> annos = getArrayAttr().getValue();
  auto it = annos.begin();
  while (it != annos.end() && !predicate(Annotation(*it)))
    ++it;

  // Fast path for sets where the predicate never matched.
  if (it == annos.end())
    return false;

  // Build a filtered list of annotations.
  SmallVector<Attribute> filteredAnnos;
  filteredAnnos.reserve(annos.size());
  filteredAnnos.append(annos.begin(), it);
  ++it;
  while (it != annos.end()) {
    if (!predicate(Annotation(*it)))
      filteredAnnos.push_back(*it);
    ++it;
  }
  annotations = ArrayAttr::get(getContext(), filteredAnnos);
  return true;
}

/// Remove all annotations from an operation for which `predicate` returns true.
bool AnnotationSet::removeAnnotations(
    Operation *op, llvm::function_ref<bool(Annotation)> predicate) {
  AnnotationSet annos(op);
  if (!annos.empty() && annos.removeAnnotations(predicate)) {
    annos.applyToOperation(op);
    return true;
  }
  return false;
}

bool AnnotationSet::removeAnnotations(Operation *op, StringRef className) {
  return removeAnnotations(
      op, [&](Annotation a) { return (a.getClass() == className); });
}

/// Remove all port annotations from a module or extmodule for which `predicate`
/// returns true.
bool AnnotationSet::removePortAnnotations(
    Operation *module,
    llvm::function_ref<bool(unsigned, Annotation)> predicate) {
  auto ports = module->getAttr("portAnnotations").dyn_cast_or_null<ArrayAttr>();
  if (!ports || ports.empty())
    return false;

  // Collect results
  SmallVector<Attribute> newAnnos;

  // Filter the annotations on each argument.
  bool changed = false;
  for (unsigned argNum = 0, argNumEnd = ports.size(); argNum < argNumEnd;
       ++argNum) {
    AnnotationSet annos(AnnotationSet(cast<ArrayAttr>(ports[argNum])));

    // Go through all annotations on this port and extract the interesting
    // ones. If any modifications were done, keep a reduced set of attributes
    // around for the port, otherwise just stick with the existing ones.
    if (!annos.empty())
      changed |= annos.removeAnnotations(
          [&](Annotation anno) { return predicate(argNum, anno); });
    newAnnos.push_back(annos.getArrayAttr());
  }

  // If we have made any changes, apply them to the operation.
  if (changed)
    module->setAttr("portAnnotations",
                    ArrayAttr::get(module->getContext(), newAnnos));
  return changed;
}

//===----------------------------------------------------------------------===//
// Annotation
//===----------------------------------------------------------------------===//

DictionaryAttr Annotation::getDict() const {
  return cast<DictionaryAttr>(attr);
}

void Annotation::setDict(DictionaryAttr dict) { attr = dict; }

unsigned Annotation::getFieldID() const {
  if (auto fieldID = getMember<IntegerAttr>("circt.fieldID"))
    return fieldID.getInt();
  return 0;
}

/// Return the 'class' that this annotation is representing.
StringAttr Annotation::getClassAttr() const {
  return getDict().getAs<StringAttr>("class");
}

/// Return the 'class' that this annotation is representing.
StringRef Annotation::getClass() const {
  if (auto classAttr = getClassAttr())
    return classAttr.getValue();
  return {};
}

void Annotation::setMember(StringAttr name, Attribute value) {
  setMember(name.getValue(), value);
}

void Annotation::setMember(StringRef name, Attribute value) {
  // Binary search for the matching field.
  auto dict = getDict();
  auto [it, found] = mlir::impl::findAttrSorted(dict.begin(), dict.end(), name);
  auto index = std::distance(dict.begin(), it);
  // Create an array for the new members.
  SmallVector<NamedAttribute> attributes;
  attributes.reserve(dict.size() + 1);
  // Copy over the leading annotations.
  for (auto field : dict.getValue().take_front(index))
    attributes.push_back(field);
  // Push the new member.
  auto nameAttr = StringAttr::get(dict.getContext(), name);
  attributes.push_back(NamedAttribute(nameAttr, value));
  // Copy remaining members, skipping the old field value.
  for (auto field : dict.getValue().drop_front(index + found))
    attributes.push_back(field);
  // Commit the dictionary.
  setDict(DictionaryAttr::getWithSorted(dict.getContext(), attributes));
}

void Annotation::removeMember(StringAttr name) {
  auto dict = getDict();
  SmallVector<NamedAttribute> attributes;
  attributes.reserve(dict.size() - 1);
  auto i = dict.begin();
  auto e = dict.end();
  while (i != e && i->getValue() != name)
    attributes.push_back(*(i++));
  // If the member was not here, just return.
  if (i == e)
    return;
  // Copy the rest of the members over.
  attributes.append(++i, e);
  // Commit the dictionary.
  setDict(DictionaryAttr::getWithSorted(dict.getContext(), attributes));
}

void Annotation::removeMember(StringRef name) {
  // Binary search for the matching field.
  auto dict = getDict();
  auto [it, found] = mlir::impl::findAttrSorted(dict.begin(), dict.end(), name);
  auto index = std::distance(dict.begin(), it);
  if (!found)
    return;
  // Create an array for the new members.
  SmallVector<NamedAttribute> attributes;
  attributes.reserve(dict.size() - 1);
  // Copy over the leading annotations.
  for (auto field : dict.getValue().take_front(index))
    attributes.push_back(field);
  // Copy remaining members, skipping the old field value.
  for (auto field : dict.getValue().drop_front(index + 1))
    attributes.push_back(field);
  // Commit the dictionary.
  setDict(DictionaryAttr::getWithSorted(dict.getContext(), attributes));
}

void Annotation::dump() { attr.dump(); }

//===----------------------------------------------------------------------===//
// AnnotationSetIterator
//===----------------------------------------------------------------------===//

Annotation AnnotationSetIterator::operator*() const {
  return Annotation(this->getBase().getArray()[this->getIndex()]);
}

//===----------------------------------------------------------------------===//
// AnnoTarget
//===----------------------------------------------------------------------===//

FModuleLike AnnoTarget::getModule() const {
  auto *op = getOp();
  if (auto module = llvm::dyn_cast<FModuleLike>(op))
    return module;
  return op->getParentOfType<FModuleLike>();
}

AnnotationSet AnnoTarget::getAnnotations() const {
  return TypeSwitch<AnnoTarget, AnnotationSet>(*this)
      .Case<OpAnnoTarget, PortAnnoTarget>(
          [&](auto target) { return target.getAnnotations(); })
      .Default([&](auto target) { return AnnotationSet(getOp()); });
}

void AnnoTarget::setAnnotations(AnnotationSet annotations) const {
  TypeSwitch<AnnoTarget>(*this).Case<OpAnnoTarget, PortAnnoTarget>(
      [&](auto target) { target.setAnnotations(annotations); });
}

StringAttr AnnoTarget::getInnerSym(ModuleNamespace &moduleNamespace) const {
  return TypeSwitch<AnnoTarget, StringAttr>(*this)
      .Case<OpAnnoTarget, PortAnnoTarget>(
          [&](auto target) { return target.getInnerSym(moduleNamespace); })
      .Default([](auto target) { return StringAttr(); });
}

Attribute AnnoTarget::getNLAReference(ModuleNamespace &moduleNamespace) const {
  return TypeSwitch<AnnoTarget, Attribute>(*this)
      .Case<OpAnnoTarget, PortAnnoTarget>(
          [&](auto target) { return target.getNLAReference(moduleNamespace); })
      .Default([](auto target) { return Attribute(); });
}

FIRRTLType AnnoTarget::getType() const {
  return TypeSwitch<AnnoTarget, FIRRTLType>(*this)
      .Case<OpAnnoTarget, PortAnnoTarget>(
          [](auto target) { return target.getType(); })
      .Default([](auto target) { return FIRRTLType(); });
}

AnnotationSet OpAnnoTarget::getAnnotations() const {
  return AnnotationSet(getOp());
}

void OpAnnoTarget::setAnnotations(AnnotationSet annotations) const {
  annotations.applyToOperation(getOp());
}

StringAttr OpAnnoTarget::getInnerSym(ModuleNamespace &moduleNamespace) const {
  return ::getOrAddInnerSym(getOp(),
                            [&moduleNamespace](FModuleOp) -> ModuleNamespace & {
                              return moduleNamespace;
                            });
}

Attribute
OpAnnoTarget::getNLAReference(ModuleNamespace &moduleNamespace) const {
  // If the op is a module, just return the module name.
  if (auto module = llvm::dyn_cast<FModuleLike>(getOp())) {
    assert(module.getModuleNameAttr() && "invalid NLA reference");
    return FlatSymbolRefAttr::get(module.getModuleNameAttr());
  }
  // Return an inner-ref to the target.
  return ::getInnerRefTo(getOp(),
                         [&moduleNamespace](FModuleOp) -> ModuleNamespace & {
                           return moduleNamespace;
                         });
}

FIRRTLType OpAnnoTarget::getType() const {
  auto *op = getOp();
  // Annotations that target operations are resolved like inner symbols.
  if (auto is = llvm::dyn_cast<hw::InnerSymbolOpInterface>(op)) {
    auto result = is.getTargetResult();
    if (!result)
      return {};
    return type_cast<FIRRTLType>(result.getType());
  }
  // Fallback to assuming the single result is the target.
  if (op->getNumResults() != 1)
    return {};
  return type_cast<FIRRTLType>(op->getResult(0).getType());
}

PortAnnoTarget::PortAnnoTarget(FModuleLike op, unsigned portNo)
    : AnnoTarget({op, portNo}) {}

PortAnnoTarget::PortAnnoTarget(MemOp op, unsigned portNo)
    : AnnoTarget({op, portNo}) {}

AnnotationSet PortAnnoTarget::getAnnotations() const {
  if (auto memOp = llvm::dyn_cast<MemOp>(getOp()))
    return AnnotationSet::forPort(memOp, getPortNo());
  if (auto moduleOp = llvm::dyn_cast<FModuleLike>(getOp()))
    return AnnotationSet::forPort(moduleOp, getPortNo());
  llvm_unreachable("unknown port target");
  return AnnotationSet(getOp()->getContext());
}

void PortAnnoTarget::setAnnotations(AnnotationSet annotations) const {
  if (auto memOp = llvm::dyn_cast<MemOp>(getOp()))
    annotations.applyToPort(memOp, getPortNo());
  else if (auto moduleOp = llvm::dyn_cast<FModuleLike>(getOp()))
    annotations.applyToPort(moduleOp, getPortNo());
  else
    llvm_unreachable("unknown port target");
}

StringAttr PortAnnoTarget::getInnerSym(ModuleNamespace &moduleNamespace) const {
  // If this is not a module, we just need to get an inner_sym on the operation
  // itself.
  auto module = llvm::dyn_cast<FModuleLike>(getOp());
  auto target = module ? hw::InnerSymTarget(getPortNo(), module)
                       : hw::InnerSymTarget(getOp());
  return ::getOrAddInnerSym(
      target, [&moduleNamespace](FModuleLike) -> ModuleNamespace & {
        return moduleNamespace;
      });
}

Attribute
PortAnnoTarget::getNLAReference(ModuleNamespace &moduleNamespace) const {
  auto module = llvm::dyn_cast<FModuleLike>(getOp());
  auto target = module ? hw::InnerSymTarget(getPortNo(), module)
                       : hw::InnerSymTarget(getOp());
  return ::getInnerRefTo(target,
                         [&moduleNamespace](FModuleOp) -> ModuleNamespace & {
                           return moduleNamespace;
                         });
}

FIRRTLType PortAnnoTarget::getType() const {
  auto *op = getOp();
  if (auto module = llvm::dyn_cast<FModuleLike>(op))
    return type_cast<FIRRTLType>(module.getPortType(getPortNo()));
  if (llvm::isa<MemOp, InstanceOp>(op))
    return type_cast<FIRRTLType>(op->getResult(getPortNo()).getType());
  llvm_unreachable("unknow operation kind");
  return {};
}

//===----------------------------------------------------------------------===//
// Annotation Details
//===----------------------------------------------------------------------===//

/// Check if an OMIR type is a string-encoded value that the FIRRTL dialect
/// simply passes through as a string without any decoding.
bool circt::firrtl::isOMIRStringEncodedPassthrough(StringRef type) {
  return type == "OMID" || type == "OMReference" || type == "OMBigInt" ||
         type == "OMLong" || type == "OMString" || type == "OMDouble" ||
         type == "OMBigDecimal" || type == "OMDeleted" || type == "OMConstant";
}

//===----------------------------------------------------------------------===//
// Utilities for Specific Annotations
//
// TODO: Remove these in favor of first-class annotations.
//===----------------------------------------------------------------------===//

LogicalResult circt::firrtl::extractDUT(const FModuleOp mod, FModuleOp &dut) {
  if (!AnnotationSet(mod).hasAnnotation(dutAnnoClass))
    return success();

  // TODO: This check is duplicated multiple places, e.g., in
  // WireDFT.  This should be factored out as part of the annotation
  // lowering pass.
  if (dut) {
    auto diag = emitError(mod->getLoc())
                << "is marked with a '" << dutAnnoClass << "', but '"
                << dut.getModuleName()
                << "' also had such an annotation (this should "
                   "be impossible!)";
    diag.attachNote(dut.getLoc()) << "the first DUT was found here";
    return failure();
  }
  dut = mod;
  return success();
}
