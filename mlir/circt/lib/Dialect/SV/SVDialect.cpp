//===- SVDialect.cpp - Implement the SV dialect ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SV dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ManagedStatic.h"

using namespace circt;
using namespace circt::sv;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void SVDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/SV/SV.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/SV/SVDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Name conflict resolution
//===----------------------------------------------------------------------===//

/// Return a StringSet that contains all of the reserved names (e.g. Verilog
/// keywords) that we need to avoid for fear of name conflicts.
struct ReservedWordsCreator {
  static void *call() {
    auto set = std::make_unique<StringSet<>>();
    static const char *const reservedWords[] = {
#include "ReservedWords.def"
    };
    for (auto *word : reservedWords)
      set->insert(word);
    return set.release();
  }
};

/// A StringSet that contains all of the reserved names (e.g., Verilog and VHDL
/// keywords) that we need to avoid to prevent naming conflicts.
static llvm::ManagedStatic<StringSet<>, ReservedWordsCreator> reservedWords;

/// Given string \p origName, generate a new name if it conflicts with any
/// keyword or any other name in the set \p recordNames. Use the int \p
/// nextGeneratedNameID as a counter for suffix. Update the \p recordNames with
/// the generated name and return the StringRef.
StringRef circt::sv::resolveKeywordConflict(
    StringRef origName, llvm::StringMap<size_t> &nextGeneratedNameIDs) {
  // Get the list of reserved words we need to avoid.  We could prepopulate this
  // into the used words cache, but it is large and immutable, so we just query
  // it when needed.

  // Fast path: name is valid
  if (!reservedWords->count(origName)) {
    auto itAndInserted = nextGeneratedNameIDs.insert({origName, 0});
    if (itAndInserted.second)
      return itAndInserted.first->getKey();
  }

  // We need to mutate the name, get the copy ready.
  SmallString<16> nameBuffer(origName.begin(), origName.end());
  nameBuffer.push_back('_');
  auto baseSize = nameBuffer.size();
  auto &nextGeneratedNameID = nextGeneratedNameIDs[origName];

  while (1) {
    // We need to auto-unique it.
    auto suffix = llvm::utostr(nextGeneratedNameID++);
    nameBuffer.append(suffix.begin(), suffix.end());

    // The name may be unique.  No keywords have an underscore followed by a
    // number, so don't check that again.
    auto itAndInserted = nextGeneratedNameIDs.insert({nameBuffer, 0});
    if (itAndInserted.second)
      return itAndInserted.first->getKey();

    // Chop off the suffix and try again until we get a unique name..
    nameBuffer.resize(baseSize);
  }
}

static bool isValidVerilogCharacterFirst(char ch) {
  return llvm::isAlpha(ch) || ch == '_';
}

static bool isValidVerilogCharacter(char ch) {
  return isValidVerilogCharacterFirst(ch) || llvm::isDigit(ch) || ch == '$';
}

/// Legalize the specified name for use in SV output. Auto-uniquifies the name
/// through \c resolveKeywordConflict if required. If the name is empty, a
/// unique temp name is created.
StringRef
circt::sv::legalizeName(StringRef name,
                        llvm::StringMap<size_t> &nextGeneratedNameIDs) {
  // Fastest path: empty name.
  if (name.empty())
    return resolveKeywordConflict("_GEN", nextGeneratedNameIDs);

  // Check that the name is valid as the semi-fast path.
  if (llvm::all_of(name, isValidVerilogCharacter) &&
      isValidVerilogCharacterFirst(name.front()))
    return resolveKeywordConflict(name, nextGeneratedNameIDs);

  // The name consists of at least one invalid character.  Escape it.
  SmallString<16> tmpName;
  if (!isValidVerilogCharacterFirst(name.front()) && name.front() != ' ' &&
      name.front() != '.')
    tmpName += '_';
  for (char ch : name) {
    if (isValidVerilogCharacter(ch))
      tmpName += ch;
    else if (ch == ' ' || ch == '.')
      tmpName += '_';
    else {
      tmpName += llvm::utohexstr((unsigned char)ch);
    }
  }

  // Make sure the new valid name does not conflict with any existing names.
  return resolveKeywordConflict(tmpName, nextGeneratedNameIDs);
}

/// Check if a name is valid for use in SV output by only containing characters
/// allowed in SV identifiers.
///
/// Call \c legalizeName() to obtain a legalized version of the name.
bool circt::sv::isNameValid(StringRef name) {
  if (name.empty())
    return false;
  if (!isValidVerilogCharacterFirst(name.front()))
    return false;
  for (char ch : name) {
    if (!isValidVerilogCharacter(ch))
      return false;
  }
  return reservedWords->count(name) == 0;
}
