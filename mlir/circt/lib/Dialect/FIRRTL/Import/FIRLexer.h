//===- FIRLexer.h - .fir lexer and token definitions ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the a Lexer and Token interface for .fir files.
//
//===----------------------------------------------------------------------===//

#ifndef FIRTOMLIR_FIRLEXER_H
#define FIRTOMLIR_FIRLEXER_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
class MLIRContext;
class Location;
} // namespace mlir

namespace circt {
namespace firrtl {

/// This represents a specific token for .fir files.
class FIRToken {
public:
  enum Kind {
#define TOK_MARKER(NAME) NAME,
#define TOK_IDENTIFIER(NAME) NAME,
#define TOK_LITERAL(NAME) NAME,
#define TOK_PUNCTUATION(NAME, SPELLING) NAME,
#define TOK_KEYWORD(SPELLING) kw_##SPELLING,
#define TOK_LPKEYWORD(SPELLING) lp_##SPELLING,
#include "FIRTokenKinds.def"
  };

  FIRToken(Kind kind, StringRef spelling) : kind(kind), spelling(spelling) {}

  // Return the bytes that make up this token.
  StringRef getSpelling() const { return spelling; }

  // Token classification.
  Kind getKind() const { return kind; }
  bool is(Kind K) const { return kind == K; }

  bool isAny(Kind k1, Kind k2) const { return is(k1) || is(k2); }

  /// Return true if this token is one of the specified kinds.
  template <typename... T>
  bool isAny(Kind k1, Kind k2, Kind k3, T... others) const {
    if (is(k1))
      return true;
    return isAny(k2, k3, others...);
  }

  bool isNot(Kind k) const { return kind != k; }

  /// Return true if this token isn't one of the specified kinds.
  template <typename... T>
  bool isNot(Kind k1, Kind k2, T... others) const {
    return !isAny(k1, k2, others...);
  }

  /// Return true if this is one of the keyword token kinds (e.g. kw_wire).
  bool isKeyword() const;

  /// Given a token containing a string literal, return its value, including
  /// removing the quote characters and unescaping the contents of the string.
  /// The lexer has already verified that this token is valid.
  std::string getStringValue() const;
  static std::string getStringValue(StringRef spelling);

  /// Given a token containing a raw string, return its value, including removing
  /// the quote characters and unescaping the quotes of the string. The lexer has
  /// already verified that this token is valid.
  std::string getRawStringValue() const;
  static std::string getRawStringValue(StringRef spelling);

  // Location processing.
  llvm::SMLoc getLoc() const;
  llvm::SMLoc getEndLoc() const;
  llvm::SMRange getLocRange() const;

private:
  /// Discriminator that indicates the sort of token this is.
  Kind kind;

  /// A reference to the entire token contents; this is always a pointer into
  /// a memory buffer owned by the source manager.
  StringRef spelling;
};

class FIRLexerCursor;

/// This implements a lexer for .fir files.
class FIRLexer {
public:
  FIRLexer(const llvm::SourceMgr &sourceMgr, mlir::MLIRContext *context);

  const llvm::SourceMgr &getSourceMgr() const { return sourceMgr; }

  /// Move to the next valid token.
  void lexToken() { curToken = lexTokenImpl(); }

  const FIRToken &getToken() const { return curToken; }

  mlir::Location translateLocation(llvm::SMLoc loc);

  /// Return the indentation level of the specified token or None if this token
  /// is preceded by another token on the same line.
  std::optional<unsigned> getIndentation(const FIRToken &tok) const;

  /// Get an opaque pointer into the lexer state that can be restored later.
  FIRLexerCursor getCursor() const;

private:
  FIRToken lexTokenImpl();

  // Helpers.
  FIRToken formToken(FIRToken::Kind kind, const char *tokStart) {
    return FIRToken(kind, StringRef(tokStart, curPtr - tokStart));
  }

  FIRToken emitError(const char *loc, const Twine &message);

  // Lexer implementation methods.
  FIRToken lexFileInfo(const char *tokStart);
  FIRToken lexInlineAnnotation(const char *tokStart);
  FIRToken lexIdentifierOrKeyword(const char *tokStart);
  FIRToken lexNumber(const char *tokStart);
  void skipComment();
  FIRToken lexString(const char *tokStart, bool isRaw);

  const llvm::SourceMgr &sourceMgr;
  const mlir::StringAttr bufferNameIdentifier;

  StringRef curBuffer;
  const char *curPtr;

  /// This is the next token that hasn't been consumed yet.
  FIRToken curToken;

  FIRLexer(const FIRLexer &) = delete;
  void operator=(const FIRLexer &) = delete;
  friend class FIRLexerCursor;
};

/// This is the state captured for a lexer cursor.
class FIRLexerCursor {
public:
  FIRLexerCursor(const FIRLexer &lexer)
      : state(lexer.curPtr), curToken(lexer.getToken()) {}

  void restore(FIRLexer &lexer) {
    lexer.curPtr = state;
    lexer.curToken = curToken;
  }

private:
  const char *state;
  FIRToken curToken;
};

inline FIRLexerCursor FIRLexer::getCursor() const {
  return FIRLexerCursor(*this);
}

} // namespace firrtl
} // namespace circt

#endif // FIRTOMLIR_FIRLEXER_H
