//===- FIRLexer.cpp - .fir file lexer implementation ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a .fir file lexer.
//
//===----------------------------------------------------------------------===//

#include "FIRLexer.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace firrtl;
using llvm::SMLoc;
using llvm::SMRange;
using llvm::SourceMgr;

#define isdigit(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS
#define isalpha(x) DO_NOT_USE_SLOW_CTYPE_FUNCTIONS

//===----------------------------------------------------------------------===//
// FIRToken
//===----------------------------------------------------------------------===//

SMLoc FIRToken::getLoc() const {
  return SMLoc::getFromPointer(spelling.data());
}

SMLoc FIRToken::getEndLoc() const {
  return SMLoc::getFromPointer(spelling.data() + spelling.size());
}

SMRange FIRToken::getLocRange() const { return SMRange(getLoc(), getEndLoc()); }

/// Return true if this is one of the keyword token kinds (e.g. kw_wire).
bool FIRToken::isKeyword() const {
  switch (kind) {
  default:
    return false;
#define TOK_KEYWORD(SPELLING)                                                  \
  case kw_##SPELLING:                                                          \
    return true;
#include "FIRTokenKinds.def"
  }
}

/// Given a token containing a string literal, return its value, including
/// removing the quote characters and unescaping the contents of the string. The
/// lexer has already verified that this token is valid.
std::string FIRToken::getStringValue() const {
  assert(getKind() == string);
  return getStringValue(getSpelling());
}

std::string FIRToken::getStringValue(StringRef spelling) {
  // Start by dropping the quotes.
  StringRef bytes = spelling.drop_front().drop_back();

  std::string result;
  result.reserve(bytes.size());
  for (size_t i = 0, e = bytes.size(); i != e;) {
    auto c = bytes[i++];
    if (c != '\\') {
      result.push_back(c);
      continue;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c1 = bytes[i++];
    switch (c1) {
    case '\\':
    case '"':
    case '\'':
      result.push_back(c1);
      continue;
    case 'b':
      result.push_back('\b');
      continue;
    case 'n':
      result.push_back('\n');
      continue;
    case 't':
      result.push_back('\t');
      continue;
    case 'f':
      result.push_back('\f');
      continue;
    case 'r':
      result.push_back('\r');
      continue;
      // TODO: Handle the rest of the escapes (octal and unicode).
    default:
      break;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c2 = bytes[i++];

    assert(llvm::isHexDigit(c1) && llvm::isHexDigit(c2) && "invalid escape");
    result.push_back((llvm::hexDigitValue(c1) << 4) | llvm::hexDigitValue(c2));
  }

  return result;
}

/// Given a token containing a raw string, return its value, including removing
/// the quote characters and unescaping the quotes of the string. The lexer has
/// already verified that this token is valid.
std::string FIRToken::getRawStringValue() const {
  assert(getKind() == raw_string);
  return getRawStringValue(getSpelling());
}

std::string FIRToken::getRawStringValue(StringRef spelling) {
  // Start by dropping the quotes.
  StringRef bytes = spelling.drop_front().drop_back();

  std::string result;
  result.reserve(bytes.size());
  for (size_t i = 0, e = bytes.size(); i != e;) {
    auto c = bytes[i++];
    if (c != '\\') {
      result.push_back(c);
      continue;
    }

    assert(i + 1 <= e && "invalid string should be caught by lexer");
    auto c1 = bytes[i++];
    if (c1 != '\'') {
      result.push_back(c);
    }
    result.push_back(c1);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// FIRLexer
//===----------------------------------------------------------------------===//

static StringAttr getMainBufferNameIdentifier(const llvm::SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

FIRLexer::FIRLexer(const llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
      curBuffer(
          sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID())->getBuffer()),
      curPtr(curBuffer.begin()),
      // Prime the first token.
      curToken(lexTokenImpl()) {}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location FIRLexer::translateLocation(llvm::SMLoc loc) {
  assert(loc.isValid());
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                             lineAndColumn.second);
}

/// Emit an error message and return a FIRToken::error token.
FIRToken FIRLexer::emitError(const char *loc, const Twine &message) {
  mlir::emitError(translateLocation(SMLoc::getFromPointer(loc)), message);
  return formToken(FIRToken::error, loc);
}

/// Return the indentation level of the specified token.
std::optional<unsigned> FIRLexer::getIndentation(const FIRToken &tok) const {
  // Count the number of horizontal whitespace characters before the token.
  auto *bufStart = curBuffer.begin();

  auto isHorizontalWS = [](char c) -> bool {
    return c == ' ' || c == '\t' || c == ',';
  };
  auto isVerticalWS = [](char c) -> bool {
    return c == '\n' || c == '\r' || c == '\f' || c == '\v';
  };

  unsigned indent = 0;
  const auto *ptr = (const char *)tok.getSpelling().data();
  while (ptr != bufStart && isHorizontalWS(ptr[-1]))
    --ptr, ++indent;

  // If the character we stopped at isn't the start of line, then return none.
  if (ptr != bufStart && !isVerticalWS(ptr[-1]))
    return std::nullopt;

  return indent;
}

//===----------------------------------------------------------------------===//
// Lexer Implementation Methods
//===----------------------------------------------------------------------===//

FIRToken FIRLexer::lexTokenImpl() {
  while (true) {
    const char *tokStart = curPtr;
    switch (*curPtr++) {
    default:
      // Handle identifiers.
      if (llvm::isAlpha(curPtr[-1]))
        return lexIdentifierOrKeyword(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that llvm::MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(FIRToken::eof, tokStart);

      [[fallthrough]]; // Treat as whitespace.

    case ' ':
    case '\t':
    case '\n':
    case '\r':
    case ',':
      // Handle whitespace.
      continue;

    case '`':
    case '_':
      // Handle identifiers.
      return lexIdentifierOrKeyword(tokStart);

    case '.':
      return formToken(FIRToken::period, tokStart);
    case ':':
      return formToken(FIRToken::colon, tokStart);
    case '(':
      return formToken(FIRToken::l_paren, tokStart);
    case ')':
      return formToken(FIRToken::r_paren, tokStart);
    case '{':
      if (*curPtr == '|')
        return ++curPtr, formToken(FIRToken::l_brace_bar, tokStart);
      return formToken(FIRToken::l_brace, tokStart);
    case '}':
      return formToken(FIRToken::r_brace, tokStart);
    case '[':
      return formToken(FIRToken::l_square, tokStart);
    case ']':
      return formToken(FIRToken::r_square, tokStart);
    case '<':
      if (*curPtr == '-')
        return ++curPtr, formToken(FIRToken::less_minus, tokStart);
      if (*curPtr == '=')
        return ++curPtr, formToken(FIRToken::less_equal, tokStart);
      return formToken(FIRToken::less, tokStart);
    case '>':
      return formToken(FIRToken::greater, tokStart);
    case '=':
      if (*curPtr == '>')
        return ++curPtr, formToken(FIRToken::equal_greater, tokStart);
      return formToken(FIRToken::equal, tokStart);
    case '?':
      return formToken(FIRToken::question, tokStart);
    case '@':
      if (*curPtr == '[')
        return lexFileInfo(tokStart);
      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");
    case '%':
      if (*curPtr == '[')
        return lexInlineAnnotation(tokStart);
      return emitError(tokStart, "unexpected character following '%'");
    case '|':
      if (*curPtr == '}')
        return ++curPtr, formToken(FIRToken::r_brace_bar, tokStart);
      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case ';':
      skipComment();
      continue;

    case '"':
      return lexString(tokStart, /*isRaw=*/false);
    case '\'':
      return lexString(tokStart, /*isRaw=*/true);

    case '+':
    case '-':
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexNumber(tokStart);
    }
  }
}

/// Lex a file info specifier.
///
///   FileInfo ::= '@[' ('\]'|.)* ']'
///
FIRToken FIRLexer::lexFileInfo(const char *tokStart) {
  while (1) {
    switch (*curPtr++) {
    case ']': // This is the end of the fileinfo literal.
      return formToken(FIRToken::fileinfo, tokStart);
    case '\\':
      // Ignore escaped ']'
      if (*curPtr == ']')
        ++curPtr;
      break;
    case 0:
      // This could be the end of file in the middle of the fileinfo.  If so
      // emit an error.
      if (curPtr - 1 != curBuffer.end())
        break;
      [[fallthrough]];
    case '\n': // Vertical whitespace isn't allowed in a fileinfo.
    case '\v':
    case '\f':
      return emitError(tokStart, "unterminated file info specifier");
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// Lex a non-standard inline Annotation file.
///
/// InlineAnnotation ::= '%[' (.)* ']'
///
FIRToken FIRLexer::lexInlineAnnotation(const char *tokStart) {
  size_t depth = 0;
  bool stringMode = false;
  while (1) {
    switch (*curPtr++) {
    case '\\':
      ++curPtr;
      break;
    case '"':
      stringMode = !stringMode;
      break;
    case ']':
      if (stringMode)
        break;
      if (depth == 1)
        return formToken(FIRToken::inlineannotation, tokStart);
      --depth;
      break;
    case '[':
      if (stringMode)
        break;
      ++depth;
      break;
    case 0:
      if (curPtr - 1 != curBuffer.end())
        break;
      return emitError(tokStart, "unterminated inline annotation");
    default:
      break;
    }
  }
}

/// Lex an identifier or keyword that starts with a letter.
///
///   LegalStartChar ::= [a-zA-Z_]
///   LegalIdChar    ::= LegalStartChar | [0-9] | '$'
///
///   Id ::= LegalStartChar (LegalIdChar)*
///   LiteralId ::= [a-zA-Z0-9$_]+
///
FIRToken FIRLexer::lexIdentifierOrKeyword(const char *tokStart) {
  // Remember that this is a literalID
  bool isLiteralId = *tokStart == '`';

  // Match the rest of the identifier regex: [0-9a-zA-Z_$-]*
  while (llvm::isAlpha(*curPtr) || llvm::isDigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$' || *curPtr == '-')
    ++curPtr;

  // Consume the trailing '`' in a literal identifier.
  if (isLiteralId) {
    if (*curPtr != '`')
      return emitError(tokStart, "unterminated literal identifier");
    ++curPtr;
  }

  StringRef spelling(tokStart, curPtr - tokStart);

  // Check to see if this is a 'primop', which is an identifier juxtaposed with
  // a '(' character.
  if (*curPtr == '(') {
    FIRToken::Kind kind = llvm::StringSwitch<FIRToken::Kind>(spelling)
#define TOK_LPKEYWORD(SPELLING) .Case(#SPELLING, FIRToken::lp_##SPELLING)
#include "FIRTokenKinds.def"
                              .Default(FIRToken::identifier);
    if (kind != FIRToken::identifier) {
      ++curPtr;
      return formToken(kind, tokStart);
    }
  }

  // See if the identifier is a keyword.  By default, it is an identifier.
  FIRToken::Kind kind = llvm::StringSwitch<FIRToken::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, FIRToken::kw_##SPELLING)
#include "FIRTokenKinds.def"
                            .Default(FIRToken::identifier);

  // If this has the backticks of a literal identifier and it fell through the
  // above switch, indicating that it was not found to e a keyword, then change
  // its kind from identifier to literal identifier.
  if (isLiteralId && kind == FIRToken::identifier)
    kind = FIRToken::literal_identifier;

  return FIRToken(kind, spelling);
}

/// Skip a comment line, starting with a ';' and going to end of line.
void FIRLexer::skipComment() {
  while (true) {
    switch (*curPtr++) {
    case '\n':
    case '\r':
      // Newline is end of comment.
      return;
    case 0:
      // If this is the end of the buffer, end the comment.
      if (curPtr - 1 == curBuffer.end()) {
        --curPtr;
        return;
      }
      [[fallthrough]];
    default:
      // Skip over other characters.
      break;
    }
  }
}

/// StringLit      ::= '"' UnquotedString? '"'
/// RawString      ::= '\'' UnquotedString? '\''
/// UnquotedString ::= ( '\\\'' | '\\"' | ~[\r\n] )+?
///
FIRToken FIRLexer::lexString(const char *tokStart, bool isRaw) {
  while (1) {
    switch (*curPtr++) {
    case '"': // This is the end of the string literal.
      if (isRaw)
        break;
      return formToken(FIRToken::string, tokStart);
    case '\'': // This is the end of the raw string.
      if (!isRaw)
        break;
      return formToken(FIRToken::raw_string, tokStart);
    case '\\':
      // Ignore escaped '\'' or '"'
      if (*curPtr == '\'' || *curPtr == '"')
        ++curPtr;
      else if (*curPtr == 'u' || *curPtr == 'U')
        return emitError(tokStart, "unicode escape not supported in string");
      break;
    case 0:
      // This could be the end of file in the middle of the string.  If so
      // emit an error.
      if (curPtr - 1 != curBuffer.end())
        break;
      [[fallthrough]];
    case '\n': // Vertical whitespace isn't allowed in a string.
    case '\r':
    case '\v':
    case '\f':
      return emitError(tokStart, "unterminated string");
    default:
      if (curPtr[-1] & ~0x7F)
        return emitError(tokStart, "string characters must be 7-bit ASCII");
      // Skip over other characters.
      break;
    }
  }
}

/// Lex a number literal.
///
///   UnsignedInt ::= '0' | PosInt
///   PosInt ::= [1-9] ([0-9])*
///   DoubleLit ::=
///       ( '+' | '-' )? Digit+ '.' Digit+ ( 'E' ( '+' | '-' )? Digit+ )?
///   TripleLit ::=
///       Digit+ '.' Digit+ '.' Digit+
///   Radix-specified Integer ::=
///       ( '-' )? '0' ( 'b' | 'o' | 'd' | 'h' ) LegalDigit*
///
FIRToken FIRLexer::lexNumber(const char *tokStart) {
  assert(llvm::isDigit(curPtr[-1]) || curPtr[-1] == '+' || curPtr[-1] == '-');

  // There needs to be at least one digit.
  if (!llvm::isDigit(*curPtr) && !llvm::isDigit(curPtr[-1]))
    return emitError(tokStart, "unexpected character after sign");

  // If we encounter a "b", "o", "d", or "h", this is a radix-specified integer
  // literal.  This is only supported for FIRRTL 2.4.0 or later.  This is always
  // lexed, but rejected during parsing if the version is too old.
  const char *oldPtr = curPtr;
  if (curPtr[-1] == '-' && *curPtr == '0')
    ++curPtr;
  if (curPtr[-1] == '0') {
    switch (*curPtr) {
    case 'b':
      ++curPtr;
      while (*curPtr >= '0' && *curPtr <= '1')
        ++curPtr;
      return formToken(FIRToken::radix_specified_integer, tokStart);
    case 'o':
      ++curPtr;
      while (*curPtr >= '0' && *curPtr <= '7')
        ++curPtr;
      return formToken(FIRToken::radix_specified_integer, tokStart);
    case 'd':
      ++curPtr;
      while (llvm::isDigit(*curPtr))
        ++curPtr;
      return formToken(FIRToken::radix_specified_integer, tokStart);
    case 'h':
      ++curPtr;
      while (llvm::isHexDigit(*curPtr))
        ++curPtr;
      return formToken(FIRToken::radix_specified_integer, tokStart);
    default:
      curPtr = oldPtr;
      break;
    }
  }

  while (llvm::isDigit(*curPtr))
    ++curPtr;

  // If we encounter a '.' followed by a digit, then this is a floating point
  // literal, otherwise this is an integer or negative integer.
  if (*curPtr != '.' || !llvm::isDigit(curPtr[1])) {
    if (*tokStart == '-' || *tokStart == '+')
      return formToken(FIRToken::signed_integer, tokStart);
    return formToken(FIRToken::integer, tokStart);
  }

  // Lex a floating point literal.
  curPtr += 2;
  while (llvm::isDigit(*curPtr))
    ++curPtr;

  bool hasE = false;
  if (*curPtr == 'E') {
    hasE = true;
    ++curPtr;
    if (*curPtr == '+' || *curPtr == '-')
      ++curPtr;
    while (llvm::isDigit(*curPtr))
      ++curPtr;
  }

  // If we encounter a '.' followed by a digit, again, and there was no
  // exponent, then this is a version literal.  Otherwise it is a floating point
  // literal.
  if (*curPtr != '.' || !llvm::isDigit(curPtr[1]) || hasE)
    return formToken(FIRToken::floatingpoint, tokStart);

  // Lex a version literal.
  curPtr += 2;
  while (llvm::isDigit(*curPtr))
    ++curPtr;
  return formToken(FIRToken::version, tokStart);
}
