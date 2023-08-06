//===- EmissionPrinter.cpp - EmissionPrinter implementation ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the EmissionPrinter class.
//
//===----------------------------------------------------------------------===//

#include "EmissionPrinter.h"

using namespace circt;
using namespace circt::ExportSystemC;

void EmissionPrinter::emitOp(Operation *op) {
  currentLoc = op->getLoc();
  auto patterns = opPatterns.getSpecificNativePatterns().lookup(op->getName());
  for (auto *pat : patterns) {
    if (pat->matchStatement(op)) {
      pat->emitStatement(op, *this);
      return;
    }
  }

  // Emit a placeholder to the output and an error to stderr in case no valid
  // emission pattern was found.
  mlir::emitError(op->getLoc(), "no emission pattern found for '")
      << op->getName() << "'\n";
  os << "\n<<UNSUPPORTED OPERATION (" << op->getName() << ")>>\n";
  emissionFailed = true;
}

void EmissionPrinter::emitType(Type type) {
  auto patterns =
      typePatterns.getSpecificNativePatterns().lookup(type.getTypeID());
  for (auto *pat : patterns) {
    if (pat->match(type)) {
      pat->emitType(type, *this);
      return;
    }
  }

  // Emit a placeholder to the output and an error to stderr in case no valid
  // emission pattern was found.
  mlir::emitError(currentLoc, "no emission pattern found for type ")
      << type << "\n";
  os << "<<UNSUPPORTED TYPE (" << type << ")>>";
  emissionFailed = true;
}

void EmissionPrinter::emitAttr(Attribute attr) {
  auto patterns =
      attrPatterns.getSpecificNativePatterns().lookup(attr.getTypeID());
  for (auto *pat : patterns) {
    if (pat->match(attr)) {
      pat->emitAttr(attr, *this);
      return;
    }
  }

  mlir::emitError(currentLoc, "no emission pattern found for attribute ")
      << attr << "\n";
  os << "<<UNSUPPORTED ATTRIBUTE (" << attr << ")>>";
  emissionFailed = true;
}

InlineEmitter EmissionPrinter::getInlinable(Value value) {
  auto *op = value.isa<BlockArgument>() ? value.getParentRegion()->getParentOp()
                                        : value.getDefiningOp();
  Location requestLoc = currentLoc;
  currentLoc = op->getLoc();
  auto patterns = opPatterns.getSpecificNativePatterns().lookup(op->getName());
  for (auto *pat : patterns) {
    MatchResult match = pat->matchInlinable(value);
    if (!match.failed()) {
      return InlineEmitter([=]() { pat->emitInlined(value, *this); },
                           match.getPrecedence(), *this);
    }
  }

  // Emit a placeholder to the output and an error to stderr in case no valid
  // emission pattern was found.
  emissionFailed = true;
  auto err =
      mlir::emitError(value.getLoc(), "inlining not supported for value '")
      << value << "'\n";
  err.attachNote(requestLoc) << "requested to be inlined here";
  return InlineEmitter(
      [=]() { os << "<<INVALID VALUE TO INLINE (" << value << ")>>"; },
      Precedence::LIT, *this);
}

void EmissionPrinter::emitRegion(Region &region) {
  auto scope = os.scope("{\n", "}\n");
  emitRegion(region, scope);
}

void EmissionPrinter::emitRegion(
    Region &region, mlir::raw_indented_ostream::DelimitedScope &scope) {
  assert(region.hasOneBlock() &&
         "only regions with exactly one block are supported for now");

  for (Operation &op : region.getBlocks().front()) {
    emitOp(&op);
  }
}

InFlightDiagnostic EmissionPrinter::emitError(Operation *op,
                                              const Twine &message) {
  emissionFailed = true;
  os << "<<ERROR (" << message << ")>>";
  return op->emitOpError();
}

InFlightDiagnostic EmissionPrinter::emitError(const Twine &message) {
  emissionFailed = true;
  os << "<<ERROR (" << message << ")>>";
  return mlir::emitError(currentLoc, message);
}

EmissionPrinter &EmissionPrinter::operator<<(StringRef str) {
  os << str;
  return *this;
}

EmissionPrinter &EmissionPrinter::operator<<(int64_t num) {
  os << std::to_string(num);
  return *this;
}

void InlineEmitter::emitWithParensOnLowerPrecedence(Precedence prec,
                                                    StringRef lParen,
                                                    StringRef rParen) const {
  if (precedence >= prec)
    printer << lParen;

  emitter();

  if (precedence >= prec)
    printer << rParen;
}
