//===- PrettyPrinter.h - Pretty printing ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a pretty-printer.
// "PrettyPrinting", Derek C. Oppen, 1980.
// https://dx.doi.org/10.1145/357114.357115
//
// This was selected as it is linear in number of tokens O(n) and requires
// memory O(linewidth).
//
// See PrettyPrinter.cpp for more information.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PRETTYPRINTER_H
#define CIRCT_SUPPORT_PRETTYPRINTER_H

#include "circt/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SaveAndRestore.h"

#include <cstdint>
#include <deque>
#include <limits>

namespace circt {
namespace pretty {

//===----------------------------------------------------------------------===//
// Tokens
//===----------------------------------------------------------------------===//

/// Style of breaking within a group:
/// - Consistent: all fits or all breaks.
/// - Inconsistent: best fit, break where needed.
/// - Never: force no breaking including nested groups.
enum class Breaks { Consistent, Inconsistent, Never };

/// Style of indent when starting a group:
/// - Visual: offset is relative to current column.
/// - Block: offset is relative to current base indentation.
enum class IndentStyle { Visual, Block };

class Token {
public:
  enum class Kind { String, Break, Begin, End };

  struct TokenInfo {
    Kind kind; // Common initial sequence.
  };
  struct StringInfo : public TokenInfo {
    uint32_t len;
    const char *str;
  };
  struct BreakInfo : public TokenInfo {
    uint32_t spaces; // How many spaces to emit when not broken.
    int32_t offset;  // How many spaces to emit when broken.
    bool neverbreak; // If set, behaves like break except this always 'fits'.
  };
  struct BeginInfo : public TokenInfo {
    int32_t offset; // Adjust base indentation by this amount.
    Breaks breaks;
    IndentStyle style;
  };
  struct EndInfo : public TokenInfo {
    // Nothing
  };

private:
  union {
    TokenInfo info;
    StringInfo stringInfo;
    BreakInfo breakInfo;
    BeginInfo beginInfo;
    EndInfo endInfo;
  } data;

protected:
  template <Kind k, typename T>
  static auto &getInfoImpl(T &t) {
    if constexpr (k == Kind::String)
      return t.data.stringInfo;
    if constexpr (k == Kind::Break)
      return t.data.breakInfo;
    if constexpr (k == Kind::Begin)
      return t.data.beginInfo;
    if constexpr (k == Kind::End)
      return t.data.endInfo;
    llvm_unreachable("unhandled token kind");
  }

  Token(Kind k) { data.info.kind = k; }

public:
  Kind getKind() const { return data.info.kind; }
};

/// Helper class to CRTP-derive common functions.
template <class DerivedT, Token::Kind DerivedKind>
struct TokenBase : public Token {
  static bool classof(const Token *t) { return t->getKind() == DerivedKind; }

protected:
  TokenBase() : Token(DerivedKind) {}

  using InfoType = std::remove_reference_t<std::invoke_result_t<
      decltype(Token::getInfoImpl<DerivedKind, Token &>), Token &>>;

  InfoType &getInfoMut() { return Token::getInfoImpl<DerivedKind>(*this); }

  const InfoType &getInfo() const {
    return Token::getInfoImpl<DerivedKind>(*this);
  }

  template <typename... Args>
  void initialize(Args &&...args) {
    getInfoMut() = InfoType{{DerivedKind}, args...};
  }
};

/// Token types.

struct StringToken : public TokenBase<StringToken, Token::Kind::String> {
  StringToken(llvm::StringRef text) {
    assert(text.size() == (uint32_t)text.size());
    initialize((uint32_t)text.size(), text.data());
  }
  StringRef text() const { return StringRef(getInfo().str, getInfo().len); }
};

struct BreakToken : public TokenBase<BreakToken, Token::Kind::Break> {
  BreakToken(uint32_t spaces = 1, int32_t offset = 0, bool neverbreak = false) {
    initialize(spaces, offset, neverbreak);
  }
  uint32_t spaces() const { return getInfo().spaces; }
  int32_t offset() const { return getInfo().offset; }
  bool neverbreak() const { return getInfo().neverbreak; }
};

struct BeginToken : public TokenBase<BeginToken, Token::Kind::Begin> {
  BeginToken(int32_t offset = 2, Breaks breaks = Breaks::Inconsistent,
             IndentStyle style = IndentStyle::Visual) {
    initialize(offset, breaks, style);
  }
  int32_t offset() const { return getInfo().offset; }
  Breaks breaks() const { return getInfo().breaks; }
  IndentStyle style() const { return getInfo().style; }
};

struct EndToken : public TokenBase<EndToken, Token::Kind::End> {};

//===----------------------------------------------------------------------===//
// PrettyPrinter
//===----------------------------------------------------------------------===//

class PrettyPrinter {
public:
  /// Listener to Token storage events.
  struct Listener {
    virtual ~Listener();
    /// No tokens referencing external memory are present.
    virtual void clear(){};
  };

  /// PrettyPrinter for specified stream.
  /// - margin: line width.
  /// - baseIndent: always indent at least this much (starting 'indent' value).
  /// - currentColumn: current column, used to calculate space remaining.
  /// - maxStartingIndent: max column indentation starts at, must be >= margin.
  PrettyPrinter(llvm::raw_ostream &os, uint32_t margin, uint32_t baseIndent = 0,
                uint32_t currentColumn = 0,
                uint32_t maxStartingIndent = kInfinity / 4,
                Listener *listener = nullptr)
      : space(margin - std::max(currentColumn, baseIndent)),
        defaultFrame{baseIndent, PrintBreaks::Inconsistent}, indent(baseIndent),
        margin(margin), maxStartingIndent(std::max(maxStartingIndent, margin)),
        os(os), listener(listener) {
    assert(maxStartingIndent < kInfinity / 2);
    assert(maxStartingIndent > baseIndent);
    assert(margin > currentColumn);
    // Ensure first print advances to at least baseIndent.
    pendingIndentation =
        baseIndent > currentColumn ? baseIndent - currentColumn : 0;
  }
  ~PrettyPrinter() { eof(); }

  /// Add token for printing.  In Oppen, this is "scan".
  void add(Token t);

  /// Add a range of tokens.
  template <typename R>
  void addTokens(R &&tokens) {
    // Don't invoke listener until range processed, we own it now.
    {
      llvm::SaveAndRestore<Listener *> save(listener, nullptr);
      for (Token &t : tokens)
        add(t);
    }
    // Invoke it now if appropriate.
    if (scanStack.empty())
      clear();
  }

  void eof();

  void setListener(Listener *newListener) { listener = newListener; };
  auto *getListener() const { return listener; }

  static constexpr uint32_t kInfinity = (1U << 15) - 1;

private:
  /// Format token with tracked size.
  struct FormattedToken {
    Token token;  /// underlying token
    int32_t size; /// calculate size when positive.
  };

  /// Breaking style for a printStack entry.
  /// This is "Breaks" values with extra for "Fits".
  /// Breaks::Never is "AlwaysFits" here.
  enum class PrintBreaks { Consistent, Inconsistent, AlwaysFits, Fits };

  /// Printing information for active scope, stored in printStack.
  struct PrintEntry {
    uint32_t offset;
    PrintBreaks breaks;
  };

  /// Print out tokens we know sizes for, and drop from token buffer.
  void advanceLeft();

  /// Break encountered, set sizes of begin/breaks in scanStack we now know.
  void checkStack();

  /// Check if there's enough tokens to hit width, if so print.
  /// If scan size is wider than line, it's infinity.
  void checkStream();

  /// Print a token, maintaining printStack for context.
  void print(const FormattedToken &f);

  /// Clear token buffer, scanStack must be empty.
  void clear();

  /// Reset leftTotal and tokenOffset, rebase size data and scanStack indices.
  void rebaseIfNeeded();

  /// Get current printing frame.
  auto &getPrintFrame() {
    return printStack.empty() ? defaultFrame : printStack.back();
  }

  /// Characters left on this line.
  int32_t space;

  /// Sizes: printed, enqueued
  int32_t leftTotal;
  int32_t rightTotal;

  /// Unprinted tokens, combination of 'token' and 'size' in Oppen.
  std::deque<FormattedToken> tokens;
  /// index of first token, for resolving scanStack entries.
  uint32_t tokenOffset = 0;

  /// Stack of begin/break tokens, adjust by tokenOffset to index into tokens.
  std::deque<uint32_t> scanStack;

  /// Stack of printing contexts (indentation + breaking behavior).
  SmallVector<PrintEntry> printStack;

  /// Printing context when stack is empty.
  const PrintEntry defaultFrame;

  /// Number of "AlwaysFits" on print stack.
  uint32_t alwaysFits = 0;

  /// Current indentation level
  uint32_t indent;

  /// Whitespace to print before next, tracked to avoid trailing whitespace.
  uint32_t pendingIndentation;

  /// Target line width.
  const uint32_t margin;

  /// Maximum starting indentation level (default=kInfinity/4).
  /// Useful to continue indentation past margin while still providing a limit
  /// to avoid pathological output and for consumption by tools with limits.
  const uint32_t maxStartingIndent;

  /// Output stream.
  llvm::raw_ostream &os;

  /// Hook for Token storage events.
  Listener *listener = nullptr;

  /// Threshold for walking scan state and "rebasing" totals/offsets.
  static constexpr decltype(leftTotal) rebaseThreshold =
      1UL << (std::numeric_limits<decltype(leftTotal)>::digits - 3);
};

} // end namespace pretty
} // end namespace circt

#endif // CIRCT_SUPPORT_PRETTYPRINTER_H
