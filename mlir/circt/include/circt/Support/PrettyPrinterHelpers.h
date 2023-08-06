//===- PrettyPrinterHelpers.h - Pretty printing helpers -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper classes for using PrettyPrinter.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PRETTYPRINTERHELPERS_H
#define CIRCT_SUPPORT_PRETTYPRINTERHELPERS_H

#include "circt/Support/PrettyPrinter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"

namespace circt {
namespace pretty {

//===----------------------------------------------------------------------===//
// PrettyPrinter standin that buffers tokens until flushed.
//===----------------------------------------------------------------------===//

/// Buffer tokens for clients that need to adjust things.
struct BufferingPP {
  using BufferVec = SmallVectorImpl<Token>;
  BufferVec &tokens;
  bool hasEOF = false;

  BufferingPP(BufferVec &tokens) : tokens(tokens) {}

  void add(Token t) {
    assert(!hasEOF);
    tokens.push_back(t);
  }

  /// Add a range of tokens.
  template <typename R>
  void addTokens(R &&newTokens) {
    assert(!hasEOF);
    llvm::append_range(tokens, newTokens);
  }

  /// Buffer a final EOF, no tokens allowed after this.
  void eof() {
    assert(!hasEOF);
    hasEOF = true;
  }

  /// Flush buffered tokens to the specified pretty printer.
  /// Emit the EOF is one was added.
  void flush(PrettyPrinter &pp) {
    pp.addTokens(tokens);
    tokens.clear();
    if (hasEOF) {
      pp.eof();
      hasEOF = false;
    }
  }
};

//===----------------------------------------------------------------------===//
// Convenience Token builders.
//===----------------------------------------------------------------------===//

namespace detail {
void emitNBSP(unsigned n, llvm::function_ref<void(Token)> add);
} // end namespace detail

/// Add convenience methods for generating pretty-printing tokens.
template <typename PPTy = PrettyPrinter>
class TokenBuilder {
  PPTy &pp;

public:
  TokenBuilder(PPTy &pp) : pp(pp) {}

  //===- Add tokens -------------------------------------------------------===//

  /// Add new token.
  template <typename T, typename... Args>
  typename std::enable_if_t<std::is_base_of_v<Token, T>> add(Args &&...args) {
    pp.add(T(std::forward<Args>(args)...));
  }
  void addToken(Token t) { pp.add(t); }

  /// End of a stream.
  void eof() { pp.eof(); }

  //===- Strings ----------------------------------------------------------===//

  /// Add a literal (with external storage).
  void literal(StringRef str) { add<StringToken>(str); }

  /// Add a non-breaking space.
  void nbsp() { literal(" "); }

  /// Add multiple non-breaking spaces as a single token.
  void nbsp(unsigned n) {
    detail::emitNBSP(n, [&](Token t) { addToken(t); });
  }

  //===- Breaks -----------------------------------------------------------===//

  /// Add a 'neverbreak' break.  Always 'fits'.
  void neverbreak() { add<BreakToken>(0, 0, true); }

  /// Add a newline (break too wide to fit, always breaks).
  void newline() { add<BreakToken>(PrettyPrinter::kInfinity); }

  /// Add breakable spaces.
  void spaces(uint32_t n) { add<BreakToken>(n); }

  /// Add a breakable space.
  void space() { spaces(1); }

  /// Add a break that is zero-wide if not broken.
  void zerobreak() { add<BreakToken>(0); }

  //===- Groups -----------------------------------------------------------===//

  /// Start a IndentStyle::Block group with specified offset.
  void bbox(int32_t offset = 0, Breaks breaks = Breaks::Consistent) {
    add<BeginToken>(offset, breaks, IndentStyle::Block);
  }

  /// Start a consistent group with specified offset.
  void cbox(int32_t offset = 0, IndentStyle style = IndentStyle::Visual) {
    add<BeginToken>(offset, Breaks::Consistent, style);
  }

  /// Start an inconsistent group with specified offset.
  void ibox(int32_t offset = 0, IndentStyle style = IndentStyle::Visual) {
    add<BeginToken>(offset, Breaks::Inconsistent, style);
  }

  /// Start a group that cannot break, including nested groups.
  /// Use sparingly.
  void neverbox() { add<BeginToken>(0, Breaks::Never); }

  /// End a group.
  void end() { add<EndToken>(); }
};

/// PrettyPrinter::Listener that saves strings while live.
/// Once they're no longer referenced, memory is reset.
/// Allows differentiating between strings to save and external strings.
class TokenStringSaver : public PrettyPrinter::Listener {
  llvm::BumpPtrAllocator alloc;
  llvm::StringSaver strings;

public:
  TokenStringSaver() : strings(alloc) {}

  /// Add string, save in storage.
  [[nodiscard]] StringRef save(StringRef str) { return strings.save(str); }

  /// PrettyPrinter::Listener::clear -- indicates no external refs.
  void clear() override;
};

//===----------------------------------------------------------------------===//
// Streaming support.
//===----------------------------------------------------------------------===//

/// Send one of these to TokenStream to add the corresponding token.
/// See TokenBuilder for details of each.
enum class PP {
  bbox2,
  cbox0,
  cbox2,
  end,
  eof,
  ibox0,
  ibox2,
  nbsp,
  neverbox,
  neverbreak,
  newline,
  space,
  zerobreak,
};

/// String wrapper to indicate string has external storage.
struct PPExtString {
  StringRef str;
  explicit PPExtString(StringRef str) : str(str) {}
};

/// String wrapper to indicate string needs to be saved.
struct PPSaveString {
  StringRef str;
  explicit PPSaveString(StringRef str) : str(str) {}
};

/// Wrap a PrettyPrinter with TokenBuilder features as well as operator<<'s.
/// String behavior:
/// Strings streamed as `const char *` are assumed to have external storage,
/// and StringRef's are saved until no longer needed.
/// Use PPExtString() and PPSaveString() wrappers to specify/override behavior.
template <typename PPTy = PrettyPrinter>
class TokenStream : public TokenBuilder<PPTy> {
  using Base = TokenBuilder<PPTy>;
  TokenStringSaver &saver;

public:
  /// Create a TokenStream using the specified PrettyPrinter and StringSaver
  /// storage. Strings are saved in `saver`, which is generally the listener in
  /// the PrettyPrinter, but may not be (e.g., using BufferingPP).
  TokenStream(PPTy &pp, TokenStringSaver &saver) : Base(pp), saver(saver) {}

  /// Add a string literal (external storage).
  TokenStream &operator<<(const char *s) {
    Base::literal(s);
    return *this;
  }
  /// Add a string token (saved to storage).
  TokenStream &operator<<(StringRef s) {
    Base::template add<StringToken>(saver.save(s));
    return *this;
  }

  /// String has external storage.
  TokenStream &operator<<(const PPExtString &str) {
    Base::literal(str.str);
    return *this;
  }

  /// String must be saved.
  TokenStream &operator<<(const PPSaveString &str) {
    Base::template add<StringToken>(saver.save(str.str));
    return *this;
  }

  /// Convenience for inline streaming of builder methods.
  TokenStream &operator<<(PP s) {
    switch (s) {
    case PP::bbox2:
      Base::bbox(2);
      break;
    case PP::cbox0:
      Base::cbox(0);
      break;
    case PP::cbox2:
      Base::cbox(2);
      break;
    case PP::end:
      Base::end();
      break;
    case PP::eof:
      Base::eof();
      break;
    case PP::ibox0:
      Base::ibox(0);
      break;
    case PP::ibox2:
      Base::ibox(2);
      break;
    case PP::nbsp:
      Base::nbsp();
      break;
    case PP::neverbox:
      Base::neverbox();
      break;
    case PP::neverbreak:
      Base::neverbreak();
      break;
    case PP::newline:
      Base::newline();
      break;
    case PP::space:
      Base::space();
      break;
    case PP::zerobreak:
      Base::zerobreak();
      break;
    }
    return *this;
  }

  /// Stream support for user-created Token's.
  TokenStream &operator<<(Token t) {
    Base::addToken(t);
    return *this;
  }

  /// General-purpose "format this" helper, for types not supported by
  /// operator<< yet.
  template <typename T>
  TokenStream &addAsString(T &&t) {
    invokeWithStringOS([&](auto &os) { os << std::forward<T>(t); });
    return *this;
  }

  /// Helper to invoke code with a llvm::raw_ostream argument for compatibility.
  /// All data is gathered into a single string token.
  template <typename Callable, unsigned BufferLen = 128>
  auto invokeWithStringOS(Callable &&c) {
    SmallString<BufferLen> ss;
    llvm::raw_svector_ostream ssos(ss);
    auto flush = llvm::make_scope_exit([&]() {
      if (!ss.empty())
        *this << ss;
    });
    return std::invoke(std::forward<Callable>(c), ssos);
  }

  /// Write escaped versions of the string, saved in storage.
  TokenStream &writeEscaped(StringRef str, bool useHexEscapes = false) {
    return writeQuotedEscaped(str, useHexEscapes, "", "");
  }
  TokenStream &writeQuotedEscaped(StringRef str, bool useHexEscapes = false,
                                  StringRef left = "\"",
                                  StringRef right = "\"") {
    // Add as a single StringToken.
    invokeWithStringOS([&](auto &os) {
      os << left;
      os.write_escaped(str, useHexEscapes);
      os << right;
    });
    return *this;
  }

  /// Open a box, invoke the lambda, and close it after.
  template <typename T, typename Callable>
  auto scopedBox(T &&t, Callable &&c, Token close = EndToken()) {
    *this << std::forward<T>(t);
    auto done = llvm::make_scope_exit([&]() { *this << close; });
    return std::invoke(std::forward<Callable>(c));
  }
};

} // end namespace pretty
} // end namespace circt

#endif // CIRCT_SUPPORT_PRETTYPRINTERHELPERS_H
