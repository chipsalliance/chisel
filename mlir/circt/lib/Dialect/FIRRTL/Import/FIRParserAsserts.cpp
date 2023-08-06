//===- FIRParserAsserts.cpp - Printf-encoded assert handling --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements handling of printf-encoded verification operations
// embedded in when blocks.
//
//===----------------------------------------------------------------------===//

#include "FIRAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"

using namespace circt;
using namespace firrtl;

namespace json = llvm::json;

namespace {
/// Helper class to destructure parsed JSON and emit appropriate error messages.
/// This class should be treated with the same care as a Twine; it should never
/// be assigned to a local variable and should only be passed by constant
/// reference parameters.
template <typename JsonType>
class ExtractionSummaryCursor {

  // Allow ExtractionSummaryCursor to construct other instances.
  template <typename T>
  friend class ExtractionSummaryCursor;

  // This is a friend function since we have no public contructors.
  template <typename T, typename FnType>
  friend ParseResult parseJson(Location loc, const T &jsonValue, FnType fn);

  // Private constructor and destructor.
  ExtractionSummaryCursor(Location loc, Twine path, JsonType value)
      : loc(loc), path(path), value(value) {}
  ~ExtractionSummaryCursor() = default;

  // Deleted constructors.
  ExtractionSummaryCursor(const ExtractionSummaryCursor &) = delete;
  ExtractionSummaryCursor &operator=(const ExtractionSummaryCursor &) = delete;

public:
  Location loc;
  Twine path;
  JsonType value;

  /// Report an error about the current path.
  InFlightDiagnostic emitError() const {
    auto diag = mlir::emitError(loc, "extraction summary ");
    if (!path.isTriviallyEmpty())
      diag << "field `" << path << "` ";
    return diag;
  }

  /// Access a field in an object.
  ParseResult withField(StringRef field,
                        llvm::function_ref<ParseResult(
                            const ExtractionSummaryCursor<json::Value *> &)>
                            fn,
                        bool optional = false) const {
    auto fieldValue = value->get(field);
    if (!fieldValue) {
      if (optional)
        return success();
      emitError() << "missing `" << field << "` field";
      return failure();
    }
    return fn({loc, path.isTriviallyEmpty() ? field : path + "." + field,
               fieldValue});
  }

  /// Access a JSON value as an object.
  ParseResult withObject(llvm::function_ref<ParseResult(
                             const ExtractionSummaryCursor<json::Object *> &)>
                             fn) const {
    auto obj = value->getAsObject();
    if (!obj) {
      emitError() << "must be an object";
      return failure();
    }
    return fn({loc, path, obj});
  }

  /// Access a JSON value as a string.
  ParseResult withString(llvm::function_ref<ParseResult(
                             const ExtractionSummaryCursor<StringRef> &)>
                             fn) const {
    auto str = value->getAsString();
    if (!str) {
      emitError() << "must be a string";
      return failure();
    }
    return fn({loc, path, *str});
  }

  /// Access a JSON value as an array.
  ParseResult withArray(llvm::function_ref<ParseResult(
                            const ExtractionSummaryCursor<json::Value *> &)>
                            fn) const {
    auto array = value->getAsArray();
    if (!array) {
      emitError() << "must be an array";
      return failure();
    }
    for (size_t i = 0, e = array->size(); i < e; ++i)
      if (fn({loc, path + "[" + Twine(i) + "]", &(*array)[i]}))
        return failure();
    return success();
  }

  /// Access an object field as an object.
  ParseResult
  withObjectField(StringRef field,
                  llvm::function_ref<ParseResult(
                      const ExtractionSummaryCursor<json::Object *> &)>
                      fn,
                  bool optional = false) const {
    return withField(
        field, [&](const auto &cursor) { return cursor.withObject(fn); },
        optional);
  }

  /// Access an object field as a string.
  ParseResult withStringField(StringRef field,
                              llvm::function_ref<ParseResult(
                                  const ExtractionSummaryCursor<StringRef> &)>
                                  fn,
                              bool optional = false) const {
    return withField(
        field, [&](const auto &cursor) { return cursor.withString(fn); },
        optional);
  }

  /// Access an object field as an array.
  ParseResult
  withArrayField(StringRef field,
                 llvm::function_ref<ParseResult(
                     const ExtractionSummaryCursor<json::Value *> &)>
                     fn,
                 bool optional = true) const {
    return withField(
        field, [&](const auto &cursor) { return cursor.withArray(fn); },
        optional);
  }
};

/// Convenience function to create a `ExtractionSummaryCursor`.
template <typename JsonType, typename FnType>
ParseResult parseJson(Location loc, const JsonType &jsonValue, FnType fn) {
  return fn(ExtractionSummaryCursor<JsonType>{loc, {}, jsonValue});
}
} // namespace

/// A flavor of when-printf-encoded verification statement.
enum class VerifFlavor {
  VerifLibAssert, // contains "[verif-library-assert]"
  VerifLibAssume, // contains "[verif-library-assume]"
  VerifLibCover,  // contains "[verif-library-cover]"
  Assert,         // begins with "assert:"
  Assume,         // begins with "assume:"
  Cover,          // begins with "cover:"
  ChiselAssert,   // begins with "Assertion failed"
  AssertNotX      // begins with "assertNotX:"
};

/// A modifier for an assertion predicate.
enum class PredicateModifier { NoMod, TrueOrIsX };

/// Parse a conditional compile toggle (e.g. "unrOnly") into the corresponding
/// preprocessor guard macro name (e.g. "USE_UNR_ONLY_CONSTRAINTS"), or report
/// an error.
static std::optional<StringRef>
parseConditionalCompileToggle(const ExtractionSummaryCursor<StringRef> &ex) {
  if (ex.value == "formalOnly")
    return {"USE_FORMAL_ONLY_CONSTRAINTS"};
  else if (ex.value == "unrOnly")
    return {"USE_UNR_ONLY_CONSTRAINTS"};
  ex.emitError() << "must be `formalOnly` or `unrOnly`";
  return std::nullopt;
}

/// Parse a string into a `PredicateModifier`.
static std::optional<PredicateModifier>
parsePredicateModifier(const ExtractionSummaryCursor<StringRef> &ex) {
  if (ex.value == "noMod")
    return PredicateModifier::NoMod;
  else if (ex.value == "trueOrIsX")
    return PredicateModifier::TrueOrIsX;
  ex.emitError() << "must be `noMod` or `trueOrIsX`";
  return std::nullopt;
}

/// Check that an assertion "format" is one of the admissible values, or report
/// an error.
static std::optional<StringRef>
parseAssertionFormat(const ExtractionSummaryCursor<StringRef> &ex) {
  if (ex.value == "sva" || ex.value == "ifElseFatal")
    return ex.value;
  ex.emitError() << "must be `sva` or `ifElseFatal`";
  return std::nullopt;
}

/// Chisel has a tendency to emit complex assert/assume/cover statements encoded
/// as print operations with special formatting and metadata embedded in the
/// message literal. These always reside in a when block of the following form:
///
///     when invCond:
///       printf(clock, UInt<1>(1), "...[verif-library-assert]...")
///       stop(clock, UInt<1>(1), 1)
///
/// Depending on the nature the verification operation, the `stop` may be
/// optional. The Scala implementation simply removes all `stop`s that have the
/// same condition as the printf.
ParseResult circt::firrtl::foldWhenEncodedVerifOp(PrintFOp printOp) {
  auto *context = printOp.getContext();
  auto whenStmt = dyn_cast<WhenOp>(printOp->getParentOp());

  // If the parent of printOp is not when, printOp doesn't encode a
  // verification.
  if (!whenStmt)
    return success();

  // The when blocks we're interested in don't have an else region.
  if (whenStmt.hasElseRegion())
    return success();

  // The when blocks we're interested in contain a `PrintFOp` and an optional
  // `StopOp` with the same clock and condition as the print.
  Block &thenBlock = whenStmt.getThenBlock();
  auto opIt = std::next(printOp->getIterator());
  auto opEnd = thenBlock.end();

  // optional `stop(clock, enable, ...)`
  //
  // FIXME: Currently, we can't detetct stopOp in the following IR:
  //    when invCond:
  //      printf(io.clock, UInt<1>(1), "assert: ..")
  //      stop(io.clock, UInt<1>(1), 1)
  // It is because `io.clock` will create another subfield op so StopOp is not
  // the next operation. Also, we will have to modify `stopOp.clock() !=
  // printOp.clock()` below since they are not CSEd.
  if (opIt != opEnd) {
    auto stopOp = dyn_cast<StopOp>(*opIt++);
    if (!stopOp || opIt != opEnd || stopOp.getClock() != printOp.getClock() ||
        stopOp.getCond() != printOp.getCond())
      return success();
    stopOp.erase();
  }

  // Detect if we're dealing with a verification statement, and what flavor of
  // statement it is.
  auto fmt = printOp.getFormatString();
  VerifFlavor flavor;
  if (fmt.contains("[verif-library-assert]"))
    flavor = VerifFlavor::VerifLibAssert;
  else if (fmt.contains("[verif-library-assume]"))
    flavor = VerifFlavor::VerifLibAssume;
  else if (fmt.contains("[verif-library-cover]"))
    flavor = VerifFlavor::VerifLibCover;
  else if (fmt.consume_front("assert:"))
    flavor = VerifFlavor::Assert;
  else if (fmt.consume_front("assume:"))
    flavor = VerifFlavor::Assume;
  else if (fmt.consume_front("cover:"))
    flavor = VerifFlavor::Cover;
  else if (fmt.consume_front("assertNotX:"))
    flavor = VerifFlavor::AssertNotX;
  else if (fmt.startswith("Assertion failed"))
    flavor = VerifFlavor::ChiselAssert;
  else
    return success();

  // Check if the condition of the `WhenOp` is a trivial inversion operation,
  // and remove any immediately preceding verification ops that ensure this
  // condition. This caters to the following pattern emitted by Chisel:
  //
  //     assert(clock, cond, enable, ...)
  //     node N = eq(cond, UInt<1>(0))
  //     when N:
  //       printf(clock, enable, ...)
  Value flippedCond = whenStmt.getCondition();
  if (auto node = flippedCond.getDefiningOp<NodeOp>())
    flippedCond = node.getInput();
  if (auto notOp = flippedCond.getDefiningOp<NotPrimOp>()) {
    flippedCond = notOp.getInput();
  } else if (auto eqOp = flippedCond.getDefiningOp<EQPrimOp>()) {
    auto isConst0 = [](Value v) {
      if (auto constOp = v.getDefiningOp<ConstantOp>())
        return constOp.getValue().isZero();
      return false;
    };
    if (isConst0(eqOp.getLhs()))
      flippedCond = eqOp.getRhs();
    else if (isConst0(eqOp.getRhs()))
      flippedCond = eqOp.getLhs();
    else
      flippedCond = {};
  } else {
    flippedCond = {};
  }

  // If we have found such a condition, erase any verification ops that use it
  // and that match the op we are about to assemble. This is necessary since the
  // `printf` op actually carries all the information we need for the assert,
  // while the actual `assert` has none of it. This makes me sad.
  if (flippedCond) {
    // Use a set to catch cases where a verification op is a double user of the
    // flipped condition.
    SmallPtrSet<Operation *, 1> opsToErase;
    for (const auto &user : flippedCond.getUsers()) {
      TypeSwitch<Operation *>(user).Case<AssertOp, AssumeOp, CoverOp>(
          [&](auto op) {
            if (op.getClock() == printOp.getClock() &&
                op.getEnable() == printOp.getCond() &&
                op.getPredicate() == flippedCond && !op.getIsConcurrent())
              opsToErase.insert(op);
          });
    }
    for (auto op : opsToErase)
      op->erase();
  }

  // A recursive function to move all the dependency in `printOp` operands.
  std::function<void(Operation *)> moveOperationsToParent = [&](Operation *op) {
    // If operation is not defined in when, it's ok.
    if (!op || op->getBlock() != &whenStmt.getThenBlock())
      return;

    llvm::for_each(op->getOperands(), [&](Value value) {
      moveOperationsToParent(value.getDefiningOp());
    });

    // `op` might be already moved to parent op by the previous recursive calls,
    // so check again.
    if (op->getBlock() != &whenStmt.getThenBlock())
      return;

    op->moveBefore(whenStmt);
  };

  // Move all operands in printOp to the parent block.
  llvm::for_each(printOp->getOperands(), [&](Value value) {
    moveOperationsToParent(value.getDefiningOp());
  });

  ImplicitLocOpBuilder builder(whenStmt.getLoc(), whenStmt);
  builder.setInsertionPointAfter(whenStmt);
  // CAREFUL: Since the assertions are encoded as "when something wrong, then
  // print" an error message, we're actually asserting that something is *not*
  // going wrong.
  //
  // In code:
  //
  //     when somethingGoingWrong:
  //       printf("oops")
  //
  // Actually expresses:
  //
  //     assert(not(somethingGoingWrong), "oops")
  //
  // So you'll find quite a few inversions of the when condition below to
  // represent this.

  // TODO: None of the following ops preserve interpolated operands in the
  // format string. SV allows this, and we might want to extend the
  // `firrtl.{assert,assume,cover}` operations to deal with this.

  switch (flavor) {
    // Handle the case of property verification encoded as `<assert>:<msg>` or
    // `<assert>:<label>:<msg>`.
  case VerifFlavor::Assert:
  case VerifFlavor::Assume:
  case VerifFlavor::Cover:
  case VerifFlavor::AssertNotX: {
    // Extract label and message from the format string.
    StringRef label;
    StringRef message = fmt;
    auto index = fmt.find(':');
    if (index != StringRef::npos) {
      label = fmt.slice(0, index);
      message = fmt.slice(index + 1, StringRef::npos);
    }

    // AssertNotX has the special format `assertNotX:%d:msg`, where the `%d`
    // would theoretically interpolate the value being check for X, but in
    // practice the Scala impl of ExtractTestCode just discards that `%d` label
    // and replaces it with `notX`. Also prepare the condition to be checked
    // here.
    Value predicate = whenStmt.getCondition();
    if (flavor != VerifFlavor::Cover)
      predicate = builder.create<NotPrimOp>(predicate);
    if (flavor == VerifFlavor::AssertNotX) {
      label = "notX";
      if (printOp.getSubstitutions().size() != 1) {
        printOp.emitError("printf-encoded assertNotX requires one operand");
        return failure();
      }
      // Construct a `!whenCond | (value !== 1'bx)` predicate.
      Value notCond = predicate;
      predicate = builder.create<XorRPrimOp>(printOp.getSubstitutions()[0]);
      predicate = builder.create<IsXIntrinsicOp>(predicate);
      predicate = builder.create<NotPrimOp>(predicate);
      predicate = builder.create<OrPrimOp>(notCond, predicate);
    }

    // CAVEAT: The Scala impl of ExtractTestCode explicitly sets `emitSVA` to
    // false for these flavors of verification operations. I think it's a bad
    // idea to decide at parse time if something becomes an SVA or a print+fatal
    // process, so I'm not doing this here. If we ever come across this need, we
    // may want to annotate the operation with an attribute to indicate that it
    // wants to explicitly *not* be an SVA.

    // TODO: Sanitize the label by replacing whitespace with "_" as done in the
    // Scala impl of ExtractTestCode.
    ValueRange args;
    if (printOp.getSubstitutions().size())
      args = printOp.getSubstitutions().drop_front();
    if (args.size())
      printOp.emitWarning()
          << "printf-encoded assertion has format string arguments which may "
             "cause lint warnings";
    if (flavor == VerifFlavor::Assert || flavor == VerifFlavor::AssertNotX)
      builder.create<AssertOp>(printOp.getClock(), predicate, printOp.getCond(),
                               message, args, label, true);
    else if (flavor == VerifFlavor::Assume)
      builder.create<AssumeOp>(printOp.getClock(), predicate, printOp.getCond(),
                               message, args, label, true);
    else // VerifFlavor::Cover
      builder.create<CoverOp>(printOp.getClock(), predicate, printOp.getCond(),
                              message, args, label, true);
    printOp.erase();
    break;
  }

    // Handle the case of builtin Chisel assertions.
  case VerifFlavor::ChiselAssert: {
    auto op = builder.create<AssertOp>(
        printOp.getClock(), builder.create<NotPrimOp>(whenStmt.getCondition()),
        printOp.getCond(), fmt, printOp.getSubstitutions(), "chisel3_builtin",
        true);
    op->setAttr("format", StringAttr::get(context, "ifElseFatal"));
    printOp.erase();
    break;
  }

    // Handle the SiFive verification library asserts/assumes, which contain
    // additional configuration attributes for the verification op, serialized
    // to JSON and embedded in the print message within `<extraction-summary>`
    // XML tags.
  case VerifFlavor::VerifLibAssert:
  case VerifFlavor::VerifLibAssume:
  case VerifFlavor::VerifLibCover: {
    // Isolate the JSON text in the `<extraction-summary>` XML tag.
    StringRef prefix, exStr, suffix;
    std::tie(prefix, exStr) = fmt.split("<extraction-summary>");
    std::tie(exStr, suffix) = exStr.split("</extraction-summary>");

    // The extraction summary is necessary for this kind of assertion, so we
    // throw an error if it is missing.
    if (exStr.empty()) {
      auto diag = printOp.emitError(
          "printf-encoded assert/assume requires extraction summary");
      diag.attachNote(printOp.getLoc())
          << "because printf message contains "
             "`[verif-library-{assert,assume,cover}]` tag";
      diag.attachNote(printOp.getLoc())
          << "expected JSON-encoded extraction summary in "
             "`<extraction-summary>` XML tag";
      return failure();
    }

    // Parse the extraction summary, which contains additional parameters for
    // the assertion.
    auto ex = json::parse(exStr);
    if (auto err = ex.takeError()) {
      handleAllErrors(std::move(err), [&](const json::ParseError &a) {
        printOp.emitError("failed to parse JSON extraction summary")
                .attachNote()
            << a.message();
      });
      return failure();
    }

    // The extraction summary must be an object.
    auto exObj = ex->getAsObject();
    if (!exObj) {
      printOp.emitError("extraction summary must be a JSON object");
      return failure();
    }

    // Extract and apply any predicate modifier.
    PredicateModifier predMod;
    if (parseJson(printOp.getLoc(), exObj, [&](const auto &ex) {
          return ex.withObjectField("predicateModifier", [&](const auto &ex) {
            return ex.withStringField("type", [&](const auto &ex) {
              if (auto pm = parsePredicateModifier(ex)) {
                predMod = *pm;
                return success();
              }
              return failure();
            });
          });
        }))
      return failure();

    Value predicate = whenStmt.getCondition();
    predicate = builder.create<NotPrimOp>(
        predicate); // assertion triggers when predicate fails
    switch (predMod) {
    case PredicateModifier::NoMod:
      // Leave the predicate unmodified.
      break;
    case PredicateModifier::TrueOrIsX:
      // Construct a `predicate | (^predicate === 1'bx)`.
      Value orX = builder.create<XorRPrimOp>(predicate);
      orX = builder.create<VerbatimExprOp>(UIntType::get(context, 1),
                                           "{{0}} === 1'bx", orX);
      predicate = builder.create<OrPrimOp>(predicate, orX);
      break;
    }

    // Extract the preprocessor macro names that should guard this assertion.
    SmallVector<Attribute> guards;
    if (parseJson(printOp.getLoc(), exObj, [&](const auto &ex) {
          return ex.withArrayField(
              "conditionalCompileToggles", [&](const auto &ex) {
                return ex.withObject([&](const auto &ex) {
                  return ex.withStringField("type", [&](const auto &ex) {
                    if (auto guard = parseConditionalCompileToggle(ex)) {
                      guards.push_back(
                          StringAttr::get(builder.getContext(), *guard));
                      return success();
                    }
                    return failure();
                  });
                });
              });
        }))
      return failure();

    // Extract label additions and the message.
    SmallString<32> label("verif_library");
    if (parseJson(printOp.getLoc(), exObj, [&](const auto &ex) {
          return ex.withArrayField("labelExts", [&](const auto &ex) {
            return ex.withString([&](const auto &ex) {
              label.push_back('_');
              label.append(ex.value);
              return success();
            });
          });
        }))
      return failure();

    StringRef message = fmt;
    if (parseJson(printOp.getLoc(), exObj, [&](const auto &ex) {
          return ex.withStringField("baseMsg", [&](const auto &ex) {
            message = ex.value;
            return success();
          });
        }))
      return failure();

    // Assertions carry an additional `format` field.
    std::optional<StringRef> format;
    if (flavor == VerifFlavor::VerifLibAssert) {
      if (parseJson(printOp.getLoc(), exObj, [&](const auto &ex) {
            return ex.withObjectField("format", [&](const auto &ex) {
              return ex.withStringField("type", [&](const auto &ex) {
                if (auto f = parseAssertionFormat(ex)) {
                  format = *f;
                  return success();
                }
                return failure();
              });
            });
          }))
        return failure();
    }

    // Build the verification op itself.
    Operation *op;
    // TODO: The "ifElseFatal" variant isn't actually a concurrent assertion,
    // but downstream logic assumes that isConcurrent is set.
    if (flavor == VerifFlavor::VerifLibAssert)
      op = builder.create<AssertOp>(printOp.getClock(), predicate,
                                    printOp.getCond(), message,
                                    printOp.getSubstitutions(), label, true);
    else if (flavor == VerifFlavor::VerifLibAssume)
      op = builder.create<AssumeOp>(printOp.getClock(), predicate,
                                    printOp.getCond(), message,
                                    printOp.getSubstitutions(), label, true);
    else // VerifFlavor::VerifLibCover
      op = builder.create<CoverOp>(printOp.getClock(), predicate,
                                   printOp.getCond(), message,
                                   printOp.getSubstitutions(), label, true);
    printOp.erase();

    // Attach additional attributes extracted from the JSON object.
    op->setAttr("guards", ArrayAttr::get(context, guards));
    if (format)
      op->setAttr("format", StringAttr::get(context, *format));

    break;
  }
  }

  // Clean up the `WhenOp` if it has become completely empty.
  if (thenBlock.empty())
    whenStmt.erase();
  return success();
}
