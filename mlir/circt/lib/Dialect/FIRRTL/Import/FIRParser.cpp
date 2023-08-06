//===- FIRParser.cpp - .fir to FIRRTL dialect parser ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements a .fir file parser.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRParser.h"
#include "FIRAnnotations.h"
#include "FIRLexer.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/CHIRRTLDialect.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Namespace.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/ADT/PointerEmbeddedInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

using namespace circt;
using namespace firrtl;
using namespace chirrtl;

using llvm::SMLoc;
using llvm::SourceMgr;
using mlir::LocationAttr;

namespace json = llvm::json;

//===----------------------------------------------------------------------===//
// SharedParserConstants
//===----------------------------------------------------------------------===//

namespace {

/// This class refers to immutable values and annotations maintained globally by
/// the parser which can be referred to by any active parser, even those running
/// in parallel.  This is shared by all active parsers.
struct SharedParserConstants {
  SharedParserConstants(MLIRContext *context, FIRParserOptions options)
      : context(context), options(options),
        emptyArrayAttr(ArrayAttr::get(context, {})),
        loIdentifier(StringAttr::get(context, "lo")),
        hiIdentifier(StringAttr::get(context, "hi")),
        amountIdentifier(StringAttr::get(context, "amount")),
        fieldIndexIdentifier(StringAttr::get(context, "fieldIndex")),
        indexIdentifier(StringAttr::get(context, "index")) {}

  /// The context we're parsing into.
  MLIRContext *const context;

  // Options that control the behavior of the parser.
  const FIRParserOptions options;

  /// A mapping of targets to annotations.
  /// NOTE: Clients (other than the top level Circuit parser) should not mutate
  /// this.  Do not use `annotationMap[key]`, use `aM.lookup(key)` instead.
  llvm::StringMap<ArrayAttr> annotationMap;

  /// A map from identifiers to type aliases.
  llvm::StringMap<FIRRTLType> aliasMap;

  /// An empty array attribute.
  const ArrayAttr emptyArrayAttr;

  /// Cached identifiers used in primitives.
  const StringAttr loIdentifier, hiIdentifier, amountIdentifier;
  const StringAttr fieldIndexIdentifier, indexIdentifier;

private:
  SharedParserConstants(const SharedParserConstants &) = delete;
  void operator=(const SharedParserConstants &) = delete;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// FIRParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements logic common to all levels of the parser, including
/// things like types and helper logic.
struct FIRParser {
  FIRParser(SharedParserConstants &constants, FIRLexer &lexer,
            FIRVersion &version)
      : version(version), constants(constants), lexer(lexer),
        locatorFilenameCache(constants.loIdentifier /*arbitrary non-null id*/) {
  }

  // Helper methods to get stuff from the shared parser constants.
  SharedParserConstants &getConstants() const { return constants; }
  MLIRContext *getContext() const { return constants.context; }

  FIRLexer &getLexer() { return lexer; }

  /// Return the indentation level of the specified token.
  std::optional<unsigned> getIndentation() const {
    return lexer.getIndentation(getToken());
  }

  /// Return the current token the parser is inspecting.
  const FIRToken &getToken() const { return lexer.getToken(); }
  StringRef getTokenSpelling() const { return getToken().getSpelling(); }

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  InFlightDiagnostic emitError(const Twine &message = {}) {
    return emitError(getToken().getLoc(), message);
  }
  InFlightDiagnostic emitError(SMLoc loc, const Twine &message = {});

  /// Emit a warning.
  InFlightDiagnostic emitWarning(const Twine &message = {}) {
    return emitWarning(getToken().getLoc(), message);
  }

  InFlightDiagnostic emitWarning(SMLoc loc, const Twine &message = {});

  //===--------------------------------------------------------------------===//
  // Location Handling
  //===--------------------------------------------------------------------===//

  class LocWithInfo;

  /// Encode the specified source location information into an attribute for
  /// attachment to the IR.
  Location translateLocation(llvm::SMLoc loc) {
    return lexer.translateLocation(loc);
  }

  /// Parse an @info marker if present.  If so, fill in the specified Location,
  /// if not, ignore it.
  ParseResult parseOptionalInfoLocator(LocationAttr &result);

  /// Parse an optional name that may appear in Stop, Printf, or Verification
  /// statements.
  ParseResult parseOptionalName(StringAttr &name);

  //===--------------------------------------------------------------------===//
  // Annotation Parsing
  //===--------------------------------------------------------------------===//

  /// Parse a non-standard inline Annotation JSON blob if present.  This uses
  /// the info-like encoding of %[<JSON Blob>].
  ParseResult parseOptionalAnnotations(SMLoc &loc, StringRef &result);

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(FIRToken::Kind kind) {
    if (getToken().isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  ///
  /// This returns the consumed token.
  FIRToken consumeToken() {
    FIRToken consumedToken = getToken();
    assert(consumedToken.isNot(FIRToken::eof, FIRToken::error) &&
           "shouldn't advance past EOF or errors");
    lexer.lexToken();
    return consumedToken;
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  ///
  /// This returns the consumed token.
  FIRToken consumeToken(FIRToken::Kind kind) {
    FIRToken consumedToken = getToken();
    assert(consumedToken.is(kind) && "consumed an unexpected token");
    consumeToken();
    return consumedToken;
  }

  /// Capture the current token's spelling into the specified value.  This
  /// always succeeds.
  ParseResult parseGetSpelling(StringRef &spelling) {
    spelling = getTokenSpelling();
    return success();
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(FIRToken::Kind expectedToken, const Twine &message);

  /// Parse a list of elements, terminated with an arbitrary token.
  ParseResult parseListUntil(FIRToken::Kind rightToken,
                             const std::function<ParseResult()> &parseElement);

  //===--------------------------------------------------------------------===//
  // Common Parser Rules
  //===--------------------------------------------------------------------===//

  /// Parse 'intLit' into the specified value.
  ParseResult parseIntLit(APInt &result, const Twine &message);
  ParseResult parseIntLit(int64_t &result, const Twine &message);
  ParseResult parseIntLit(int32_t &result, const Twine &message);

  // Parse 'verLit' into specified value
  ParseResult parseVersionLit(const Twine &message);

  // Parse ('<' intLit '>')? setting result to -1 if not present.
  template <typename T>
  ParseResult parseOptionalWidth(T &result);

  // Parse the 'id' grammar, which is an identifier or an allowed keyword.
  ParseResult parseId(StringRef &result, const Twine &message);
  ParseResult parseId(StringAttr &result, const Twine &message);
  ParseResult parseFieldId(StringRef &result, const Twine &message);
  ParseResult parseFieldIdSeq(SmallVectorImpl<StringRef> &result,
                              const Twine &message);
  ParseResult parseEnumType(FIRRTLType &result);
  ParseResult parseType(FIRRTLType &result, const Twine &message);

  ParseResult parseOptionalRUW(RUWAttr &result);

  /// The version of FIRRTL to use for this parser.
  FIRVersion &version;

private:
  FIRParser(const FIRParser &) = delete;
  void operator=(const FIRParser &) = delete;

  /// FIRParser is subclassed and reinstantiated.  Do not add additional
  /// non-trivial state here, add it to SharedParserConstants.
  SharedParserConstants &constants;
  FIRLexer &lexer;

  /// This is a single-entry cache for filenames in locators.
  StringAttr locatorFilenameCache;
  /// This is a single-entry cache for FileLineCol locations.
  FileLineColLoc fileLineColLocCache;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Error Handling
//===----------------------------------------------------------------------===//

InFlightDiagnostic FIRParser::emitError(SMLoc loc, const Twine &message) {
  auto diag = mlir::emitError(translateLocation(loc), message);

  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (getToken().is(FIRToken::error))
    diag.abandon();
  return diag;
}

InFlightDiagnostic FIRParser::emitWarning(SMLoc loc, const Twine &message) {
  return mlir::emitWarning(translateLocation(loc), message);
}

//===----------------------------------------------------------------------===//
// Token Parsing
//===----------------------------------------------------------------------===//

/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult FIRParser::parseToken(FIRToken::Kind expectedToken,
                                  const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitError(message);
}

/// Parse a list of elements, terminated with an arbitrary token.
ParseResult
FIRParser::parseListUntil(FIRToken::Kind rightToken,
                          const std::function<ParseResult()> &parseElement) {

  while (!consumeIf(rightToken)) {
    if (parseElement())
      return failure();
  }
  return success();
}

//===--------------------------------------------------------------------===//
// Location Processing
//===--------------------------------------------------------------------===//

/// This helper class is used to handle Info records, which specify higher level
/// symbolic source location, that may be missing from the file.  If the higher
/// level source information is missing, we fall back to the location in the
/// .fir file.
class FIRParser::LocWithInfo {
public:
  explicit LocWithInfo(SMLoc firLoc, FIRParser *parser)
      : parser(parser), firLoc(firLoc) {}

  SMLoc getFIRLoc() const { return firLoc; }

  Location getLoc() {
    if (infoLoc)
      return *infoLoc;
    auto result = parser->translateLocation(firLoc);
    infoLoc = result;
    return result;
  }

  /// Parse an @info marker if present and update our location.
  ParseResult parseOptionalInfo() {
    LocationAttr loc;
    if (failed(parser->parseOptionalInfoLocator(loc)))
      return failure();
    if (loc) {
      using ILH = FIRParserOptions::InfoLocHandling;
      switch (parser->constants.options.infoLocatorHandling) {
      case ILH::IgnoreInfo:
        assert(0 && "Should not return info locations if ignoring");
        break;
      case ILH::PreferInfo:
        infoLoc = loc;
        break;
      case ILH::FusedInfo:
        infoLoc = FusedLoc::get(loc.getContext(),
                                {loc, parser->translateLocation(firLoc)});
        break;
      }
    }
    return success();
  }

  /// If we didn't parse an info locator for the specified value, this sets a
  /// default, overriding a fall back to a location in the .fir file.
  void setDefaultLoc(Location loc) {
    if (!infoLoc)
      infoLoc = loc;
  }

private:
  FIRParser *const parser;

  /// This is the designated location in the .fir file for use when there is no
  /// @ info marker.
  SMLoc firLoc;

  /// This is the location specified by the @ marker if present.
  std::optional<Location> infoLoc;
};

/// Parse an @info marker if present.  If so, fill in the specified Location,
/// if not, ignore it.
ParseResult FIRParser::parseOptionalInfoLocator(LocationAttr &result) {
  if (getToken().isNot(FIRToken::fileinfo))
    return success();

  auto loc = getToken().getLoc();

  auto spelling = getTokenSpelling();
  consumeToken(FIRToken::fileinfo);

  auto locationPair = maybeStringToLocation(
      spelling,
      constants.options.infoLocatorHandling ==
          FIRParserOptions::InfoLocHandling::IgnoreInfo,
      locatorFilenameCache, fileLineColLocCache, getContext());

  // If parsing failed, then indicate that a weird info was found.
  if (!locationPair.first) {
    mlir::emitWarning(translateLocation(loc),
                      "ignoring unknown @ info record format");
    return success();
  }

  // If the parsing succeeded, but we are supposed to drop locators, then just
  // return.
  if (locationPair.first && constants.options.infoLocatorHandling ==
                                FIRParserOptions::InfoLocHandling::IgnoreInfo)
    return success();

  // Otherwise, set the location attribute and return.
  result = *locationPair.second;
  return success();
}

/// Parse an optional trailing name that may show up on assert, assume, cover,
/// stop, or printf.
///
/// optional_name ::= ( ':' id )?
ParseResult FIRParser::parseOptionalName(StringAttr &name) {

  if (getToken().isNot(FIRToken::colon)) {
    name = StringAttr::get(getContext(), "");
    return success();
  }

  consumeToken(FIRToken::colon);
  StringRef nameRef;
  if (parseId(nameRef, "expected result name"))
    return failure();

  name = StringAttr::get(getContext(), nameRef);

  return success();
}

//===--------------------------------------------------------------------===//
// Annotation Handling
//===--------------------------------------------------------------------===//

/// Parse a non-standard inline Annotation JSON blob if present.  This uses
/// the info-like encoding of %[<JSON Blob>].
ParseResult FIRParser::parseOptionalAnnotations(SMLoc &loc, StringRef &result) {

  if (getToken().isNot(FIRToken::inlineannotation))
    return success();

  loc = getToken().getLoc();

  result = getTokenSpelling().drop_front(2).drop_back(1);
  consumeToken(FIRToken::inlineannotation);

  return success();
}

//===--------------------------------------------------------------------===//
// Common Parser Rules
//===--------------------------------------------------------------------===//

/// intLit    ::= UnsignedInt
///           ::= SignedInt
///           ::= HexLit
///           ::= OctalLit
///           ::= BinaryLit
/// HexLit    ::= '"' 'h' ( '+' | '-' )? ( HexDigit )+ '"'
/// OctalLit  ::= '"' 'o' ( '+' | '-' )? ( OctalDigit )+ '"'
/// BinaryLit ::= '"' 'b' ( '+' | '-' )? ( BinaryDigit )+ '"'
///
ParseResult FIRParser::parseIntLit(APInt &result, const Twine &message) {
  auto spelling = getTokenSpelling();
  bool isNegative = false;
  switch (getToken().getKind()) {
  case FIRToken::signed_integer:
    isNegative = spelling[0] == '-';
    assert(spelling[0] == '+' || spelling[0] == '-');
    spelling = spelling.drop_front();
    [[fallthrough]];
  case FIRToken::integer:
    if (spelling.getAsInteger(10, result))
      return emitError(message), failure();

    // Make sure that the returned APInt has a zero at the top so clients don't
    // confuse it with a negative number.
    if (result.isNegative())
      result = result.zext(result.getBitWidth() + 1);

    if (isNegative)
      result = -result;

    // If this was parsed as >32 bits, but can be represented in 32 bits,
    // truncate off the extra width.  This is important for extmodules which
    // like parameters to be 32-bits, and insulates us from some arbitraryness
    // in StringRef::getAsInteger.
    if (result.getBitWidth() > 32 && result.getSignificantBits() <= 32)
      result = result.trunc(32);

    consumeToken();
    return success();
  case FIRToken::radix_specified_integer: {
    if (FIRVersion::compare(version, FIRVersion({2, 4, 0})) < 0)
      return emitError("Radix-specified integer literals are a FIRRTL 2.4.0 "
                       "feature, but the specified FIRRTL version was ")
             << version;
    if (spelling[0] == '-') {
      isNegative = true;
      spelling = spelling.drop_front();
    }
    unsigned base = llvm::StringSwitch<unsigned>(spelling.take_front(2))
                        .Case("0b", 2)
                        .Case("0o", 8)
                        .Case("0d", 10)
                        .Case("0h", 16);
    spelling = spelling.drop_front(2);
    if (spelling.getAsInteger(base, result))
      return emitError("invalid character in integer literal"), failure();
    if (result.isNegative())
      result = result.zext(result.getBitWidth() + 1);
    if (isNegative)
      result = -result;
    consumeToken();
    return success();
  }
  case FIRToken::string: {
    if (FIRVersion::compare(version, FIRVersion({3, 0, 0})) >= 0)
      return emitError(
          "String-encoded integer literals are unsupported after FIRRTL 3.0.0");

    // Drop the quotes.
    assert(spelling.front() == '"' && spelling.back() == '"');
    spelling = spelling.drop_back().drop_front();

    // Decode the base.
    unsigned base;
    switch (spelling.empty() ? ' ' : spelling.front()) {
    case 'h':
      base = 16;
      break;
    case 'o':
      base = 8;
      break;
    case 'b':
      base = 2;
      break;
    default:
      return emitError("expected base specifier (h/o/b) in integer literal"),
             failure();
    }
    spelling = spelling.drop_front();

    // Handle the optional sign.
    bool isNegative = false;
    if (!spelling.empty() && spelling.front() == '+')
      spelling = spelling.drop_front();
    else if (!spelling.empty() && spelling.front() == '-') {
      isNegative = true;
      spelling = spelling.drop_front();
    }

    // Parse the digits.
    if (spelling.empty())
      return emitError("expected digits in integer literal"), failure();

    if (spelling.getAsInteger(base, result))
      return emitError("invalid character in integer literal"), failure();

    // We just parsed the positive version of this number.  Make sure it has
    // a zero at the top so clients don't confuse it with a negative number and
    // so the negation (in the case of a negative sign) doesn't overflow.
    if (result.isNegative())
      result = result.zext(result.getBitWidth() + 1);

    if (isNegative)
      result = -result;

    consumeToken(FIRToken::string);
    return success();
  }

  default:
    return emitError("expected integer literal"), failure();
  }
}

ParseResult FIRParser::parseIntLit(int64_t &result, const Twine &message) {
  APInt value;
  auto loc = getToken().getLoc();
  if (parseIntLit(value, message))
    return failure();

  result = (int64_t)value.getLimitedValue(INT64_MAX);
  if (result != value)
    return emitError(loc, "value is too big to handle"), failure();
  return success();
}

ParseResult FIRParser::parseIntLit(int32_t &result, const Twine &message) {
  APInt value;
  auto loc = getToken().getLoc();
  if (parseIntLit(value, message))
    return failure();

  result = (int32_t)value.getLimitedValue(INT32_MAX);
  if (result != value)
    return emitError(loc, "value is too big to handle"), failure();
  return success();
}

/// versionLit    ::= version
/// deconstruct a version literal into parts and returns those.
ParseResult FIRParser::parseVersionLit(const Twine &message) {
  auto spelling = getTokenSpelling();
  if (getToken().getKind() != FIRToken::version)
    return emitError(message), failure();
  // form a.b.c
  auto [a, d] = spelling.split(".");
  auto [b, c] = d.split(".");
  APInt aInt, bInt, cInt;
  if (a.getAsInteger(10, aInt) || b.getAsInteger(10, bInt) ||
      c.getAsInteger(10, cInt))
    return emitError("failed to parse version string"), failure();
  version.major = aInt.getLimitedValue(UINT32_MAX);
  version.minor = bInt.getLimitedValue(UINT32_MAX);
  version.patch = cInt.getLimitedValue(UINT32_MAX);
  if (version.major != aInt || version.minor != bInt || version.patch != cInt)
    return emitError("integers out of range"), failure();
  if (FIRVersion::compare(version, FIRVersion::minimumFIRVersion()) < 0)
    return emitError() << "FIRRTL version must be >="
                       << FIRVersion::minimumFIRVersion(),
           failure();
  consumeToken(FIRToken::version);
  return success();
}

// optional-width ::= ('<' intLit '>')?
//
// This returns with result equal to -1 if not present.
template <typename T>
ParseResult FIRParser::parseOptionalWidth(T &result) {
  if (!consumeIf(FIRToken::less))
    return result = -1, success();

  // Parse a width specifier if present.
  auto widthLoc = getToken().getLoc();
  if (parseIntLit(result, "expected width") ||
      parseToken(FIRToken::greater, "expected >"))
    return failure();

  if (result < 0)
    return emitError(widthLoc, "invalid width specifier"), failure();

  return success();
}

/// id  ::= Id | keywordAsId
///
/// Parse the 'id' grammar, which is an identifier or an allowed keyword.  On
/// success, this returns the identifier in the result attribute.
ParseResult FIRParser::parseId(StringRef &result, const Twine &message) {
  switch (getToken().getKind()) {
  // The most common case is an identifier.
  case FIRToken::identifier:
  case FIRToken::literal_identifier:
// Otherwise it may be a keyword that we're allowing in an id position.
#define TOK_KEYWORD(spelling) case FIRToken::kw_##spelling:
#include "FIRTokenKinds.def"

    // Yep, this is a valid identifier or literal identifier.  Turn it into an
    // attribute.  If it is a literal identifier, then drop the leading and
    // trailing '`' (backticks).
    if (getToken().getKind() == FIRToken::literal_identifier)
      result = getTokenSpelling().drop_front().drop_back();
    else
      result = getTokenSpelling();
    consumeToken();
    return success();

  default:
    emitError(message);
    return failure();
  }
}

ParseResult FIRParser::parseId(StringAttr &result, const Twine &message) {
  StringRef name;
  if (parseId(name, message))
    return failure();

  result = StringAttr::get(getContext(), name);
  return success();
}

/// fieldId ::= Id
///         ::= RelaxedId
///         ::= UnsignedInt
///         ::= keywordAsId
///
ParseResult FIRParser::parseFieldId(StringRef &result, const Twine &message) {
  // Handle the UnsignedInt case.
  result = getTokenSpelling();
  if (consumeIf(FIRToken::integer))
    return success();

  // FIXME: Handle RelaxedId

  // Otherwise, it must be Id or keywordAsId.
  if (parseId(result, message))
    return failure();

  return success();
}

/// fieldId ::= Id
///         ::= Float
///         ::= version
///         ::= UnsignedInt
///         ::= keywordAsId
///
ParseResult FIRParser::parseFieldIdSeq(SmallVectorImpl<StringRef> &result,
                                       const Twine &message) {
  // Handle the UnsignedInt case.
  StringRef tmp = getTokenSpelling();

  if (consumeIf(FIRToken::integer)) {
    result.push_back(tmp);
    return success();
  }

  if (consumeIf(FIRToken::floatingpoint)) {
    // form a.b
    // Both a and b could have more floating point stuff, but just ignore that
    // for now.
    auto [a, b] = tmp.split(".");
    result.push_back(a);
    result.push_back(b);
    return success();
  }

  if (consumeIf(FIRToken::version)) {
    // form a.b.c
    auto [a, d] = tmp.split(".");
    auto [b, c] = d.split(".");
    result.push_back(a);
    result.push_back(b);
    result.push_back(c);
    return success();
  }

  // Otherwise, it must be Id or keywordAsId.
  if (parseId(tmp, message))
    return failure();
  result.push_back(tmp);
  return success();
}

/// enum-field ::= Id ( ':' type )? ;
/// enum-type  ::= '{|' enum-field* '|}'
ParseResult FIRParser::parseEnumType(FIRRTLType &result) {
  if (parseToken(FIRToken::l_brace_bar,
                 "expected leading '{|' in enumeration type"))
    return failure();
  SmallVector<FEnumType::EnumElement> elements;
  if (parseListUntil(FIRToken::r_brace_bar, [&]() -> ParseResult {
        auto fieldLoc = getToken().getLoc();

        // Parse the name of the tag.
        StringRef name;
        if (parseId(name, "expected valid identifier for enumeration tag"))
          return failure();

        // Parse an optional type ascription.
        FIRRTLBaseType type;
        if (consumeIf(FIRToken::colon)) {
          FIRRTLType parsedType;
          if (parseType(parsedType, "expected enumeration type"))
            return failure();
          type = type_dyn_cast<FIRRTLBaseType>(parsedType);
          if (!type)
            return emitError(fieldLoc, "field must be a base type");
        } else {
          // If there is no type specified, default to UInt<0>.
          type = UIntType::get(getContext(), 0);
        }
        elements.emplace_back(StringAttr::get(getContext(), name), type);
        return success();
      }))
    return failure();
  result = FEnumType::get(getContext(), elements);
  return success();
}

/// type ::= 'Clock'
///      ::= 'Reset'
///      ::= 'AsyncReset'
///      ::= 'UInt' optional-width
///      ::= 'SInt' optional-width
///      ::= 'Analog' optional-width
///      ::= {' field* '}'
///      ::= type '[' intLit ']'
///      ::= 'Probe' '<' type '>'
///      ::= 'RWProbe' '<' type '>'
///      ::= 'const' type
///      ::= 'String'
///      ::= id
///
/// field: 'flip'? fieldId ':' type
///
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRParser::parseType(FIRRTLType &result, const Twine &message) {
  switch (getToken().getKind()) {
  default:
    return emitError(message), failure();

  case FIRToken::kw_Clock:
    consumeToken(FIRToken::kw_Clock);
    result = ClockType::get(getContext());
    break;

  case FIRToken::kw_Reset:
    consumeToken(FIRToken::kw_Reset);
    result = ResetType::get(getContext());
    break;

  case FIRToken::kw_AsyncReset:
    consumeToken(FIRToken::kw_AsyncReset);
    result = AsyncResetType::get(getContext());
    break;

  case FIRToken::kw_UInt:
  case FIRToken::kw_SInt:
  case FIRToken::kw_Analog: {
    auto kind = getToken().getKind();
    consumeToken();

    // Parse a width specifier if present.
    int32_t width;
    if (parseOptionalWidth(width))
      return failure();

    if (kind == FIRToken::kw_SInt)
      result = SIntType::get(getContext(), width);
    else if (kind == FIRToken::kw_UInt)
      result = UIntType::get(getContext(), width);
    else {
      assert(kind == FIRToken::kw_Analog);
      result = AnalogType::get(getContext(), width);
    }
    break;
  }

  case FIRToken::kw_Probe:
  case FIRToken::kw_RWProbe: {
    auto kind = getToken().getKind();
    auto loc = getToken().getLoc();
    consumeToken();
    FIRRTLType type;

    if (parseToken(FIRToken::less, "expected '<' in reference type") ||
        parseType(type, "expected probe data type") ||
        parseToken(FIRToken::greater, "expected '>' in reference type"))
      return failure();

    bool forceable = kind == FIRToken::kw_RWProbe;

    auto innerType = type_dyn_cast<FIRRTLBaseType>(type);
    if (!innerType || innerType.containsReference())
      return emitError(loc, "cannot nest reference types");

    if (!innerType.isPassive())
      return emitError(loc, "probe inner type must be passive");

    if (forceable && innerType.containsConst())
      return emitError(loc, "rwprobe cannot contain const");

    result = RefType::get(innerType, forceable);
    break;
  }

  case FIRToken::l_brace: {
    consumeToken(FIRToken::l_brace);

    SmallVector<OpenBundleType::BundleElement, 4> elements;
    bool bundleCompatible = true;
    if (parseListUntil(FIRToken::r_brace, [&]() -> ParseResult {
          bool isFlipped = consumeIf(FIRToken::kw_flip);

          StringRef fieldName;
          FIRRTLType type;
          if (parseFieldId(fieldName, "expected bundle field name") ||
              parseToken(FIRToken::colon, "expected ':' in bundle"))
            return failure();
          auto loc = getToken().getLoc();
          if (parseType(type, "expected bundle field type"))
            return failure();

          // We require that elements of aggregates themselves
          // support notion of FieldID, reject if the type does not.
          if (!isa<hw::FieldIDTypeInterface>(type))
            return emitError(loc, "type ")
                   << type << " cannot be used as field in a bundle";

          elements.push_back(
              {StringAttr::get(getContext(), fieldName), isFlipped, type});
          bundleCompatible &= isa<BundleType::ElementType>(type);
          return success();
        }))
      return failure();

    // Try to emit base-only bundle.
    if (bundleCompatible) {
      auto bundleElements = llvm::map_range(elements, [](auto element) {
        return BundleType::BundleElement{
            element.name, element.isFlip,
            cast<BundleType::ElementType>(element.type)};
      });
      result = BundleType::get(getContext(), llvm::to_vector(bundleElements));
    } else
      result = OpenBundleType::get(getContext(), elements);
    break;
  }

  case FIRToken::l_brace_bar: {
    if (parseEnumType(result))
      return failure();
    break;
  }

  case FIRToken::identifier: {
    StringRef id;
    auto loc = getToken().getLoc();
    if (parseId(id, "expected a type alias name"))
      return failure();
    auto it = constants.aliasMap.find(id);
    if (it == constants.aliasMap.end()) {
      emitError(loc) << "type identifier `" << id << "` is not declared";
      return failure();
    }
    result = it->second;
    break;
  }
  case FIRToken::kw_const: {
    consumeToken(FIRToken::kw_const);
    auto nextToken = getToken();
    auto loc = nextToken.getLoc();

    // Guard against multiple 'const' specifications
    if (nextToken.is(FIRToken::kw_const))
      return emitError(loc, "'const' can only be specified once on a type");

    if (failed(parseType(result, message)))
      return failure();

    auto baseType = type_dyn_cast<FIRRTLBaseType>(result);
    if (!baseType)
      return emitError(loc, "only hardware types can be 'const'");

    result = baseType.getConstType(true);
    return success();
  }

  case FIRToken::kw_String:
    if (FIRVersion::compare(version, FIRVersion({3, 1, 0})) < 0)
      return emitError() << "unexpected token: Properties are a FIRRTL 3.1.0+ "
                            "feature, but the specified FIRRTL version was "
                         << version;
    consumeToken(FIRToken::kw_String);
    result = StringType::get(getContext());
    break;
  case FIRToken::kw_Integer:
    if (FIRVersion::compare(version, FIRVersion({3, 1, 0})) < 0)
      return emitError() << "unexpected token: Integers are a FIRRTL 3.1.0+ "
                            "feature, but the specified FIRRTL version was "
                         << version;
    consumeToken(FIRToken::kw_Integer);
    result = FIntegerType::get(getContext());
    break;
  case FIRToken::kw_Path:
    if (FIRVersion::compare(version, FIRVersion({3, 1, 0})) < 0)
      return emitError() << "unexpected token: Properties are a FIRRTL 3.1.0+ "
                            "feature, but the specified FIRRTL version was "
                         << version;
    consumeToken(FIRToken::kw_Path);
    result = PathType::get(getContext());
    break;
  }

  // Handle postfix vector sizes.
  while (consumeIf(FIRToken::l_square)) {
    auto sizeLoc = getToken().getLoc();
    int64_t size;
    if (parseIntLit(size, "expected width") ||
        parseToken(FIRToken::r_square, "expected ]"))
      return failure();

    if (size < 0)
      return emitError(sizeLoc, "invalid size specifier"), failure();

    // We require that elements of aggregates themselves
    // support notion of FieldID, reject if the type does not.
    if (!isa<hw::FieldIDTypeInterface>(result))
      return emitError(sizeLoc, "type ")
             << result << " cannot be used in a vector";

    auto baseType = type_dyn_cast<FIRRTLBaseType>(result);
    if (baseType)
      result = FVectorType::get(baseType, size);
    else
      result = OpenVectorType::get(result, size);
  }

  return success();
}

/// ruw ::= 'old' | 'new' | 'undefined'
ParseResult FIRParser::parseOptionalRUW(RUWAttr &result) {
  switch (getToken().getKind()) {
  default:
    break;

  case FIRToken::kw_old:
    result = RUWAttr::Old;
    consumeToken(FIRToken::kw_old);
    break;
  case FIRToken::kw_new:
    result = RUWAttr::New;
    consumeToken(FIRToken::kw_new);
    break;
  case FIRToken::kw_undefined:
    result = RUWAttr::Undefined;
    consumeToken(FIRToken::kw_undefined);
    break;
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FIRModuleContext
//===----------------------------------------------------------------------===//

// Entries in a symbol table are either an mlir::Value for the operation that
// defines the value or an unbundled ID tracking the index in the
// UnbundledValues list.
using UnbundledID = llvm::PointerEmbeddedInt<unsigned, 31>;
using SymbolValueEntry = llvm::PointerUnion<Value, UnbundledID>;

using ModuleSymbolTable =
    llvm::StringMap<std::pair<SMLoc, SymbolValueEntry>, llvm::BumpPtrAllocator>;
using ModuleSymbolTableEntry = ModuleSymbolTable::MapEntryTy;

using UnbundledValueEntry = SmallVector<std::pair<Attribute, Value>>;
using UnbundledValuesList = std::vector<UnbundledValueEntry>;
namespace {
/// This structure is used to track which entries are added while inside a scope
/// and remove them upon exiting the scope.
struct UnbundledValueRestorer {
  UnbundledValuesList &list;
  size_t startingSize;
  UnbundledValueRestorer(UnbundledValuesList &list) : list(list) {
    startingSize = list.size();
  }
  ~UnbundledValueRestorer() { list.resize(startingSize); }
};
} // namespace

using SubaccessCache = llvm::DenseMap<std::pair<Value, unsigned>, Value>;

namespace {
/// This struct provides context information that is global to the module we're
/// currently parsing into.
struct FIRModuleContext : public FIRParser {
  explicit FIRModuleContext(SharedParserConstants &constants, FIRLexer &lexer,
                            FIRVersion &version)
      : FIRParser(constants, lexer, version) {}

  // The expression-oriented nature of firrtl syntax produces tons of constant
  // nodes which are obviously redundant.  Instead of literally producing them
  // in the parser, do an implicit CSE to reduce parse time and silliness in the
  // resulting IR.
  llvm::DenseMap<std::pair<Attribute, Type>, Value> constantCache;

  /// Get a cached constant.
  Value getCachedConstantInt(ImplicitLocOpBuilder &builder, Attribute attr,
                             IntType type, APInt &value) {
    auto &result = constantCache[{attr, type}];
    if (result)
      return result;

    // Make sure to insert constants at the top level of the module to maintain
    // dominance.
    OpBuilder::InsertPoint savedIP;

    auto *parentOp = builder.getInsertionBlock()->getParentOp();
    if (!isa<FModuleOp>(parentOp)) {
      savedIP = builder.saveInsertionPoint();
      while (!isa<FModuleOp>(parentOp)) {
        builder.setInsertionPoint(parentOp);
        parentOp = builder.getInsertionBlock()->getParentOp();
      }
    }

    result = builder.create<ConstantOp>(type, value);

    if (savedIP.isSet())
      builder.setInsertionPoint(savedIP.getBlock(), savedIP.getPoint());

    return result;
  }

  //===--------------------------------------------------------------------===//
  // SubaccessCache

  /// This returns a reference with the assumption that the caller will fill in
  /// the cached value. We keep track of inserted subaccesses so that we can
  /// remove them when we exit a scope.
  Value &getCachedSubaccess(Value value, unsigned index) {
    auto &result = subaccessCache[{value, index}];
    if (!result) {
      // The outer most block won't be in the map.
      auto it = scopeMap.find(value.getParentBlock());
      if (it != scopeMap.end())
        it->second->scopedSubaccesses.push_back({result, index});
    }
    return result;
  }

  //===--------------------------------------------------------------------===//
  // SymbolTable

  /// Add a symbol entry with the specified name, returning failure if the name
  /// is already defined.
  ParseResult addSymbolEntry(StringRef name, SymbolValueEntry entry, SMLoc loc,
                             bool insertNameIntoGlobalScope = false);
  ParseResult addSymbolEntry(StringRef name, Value value, SMLoc loc,
                             bool insertNameIntoGlobalScope = false) {
    return addSymbolEntry(name, SymbolValueEntry(value), loc,
                          insertNameIntoGlobalScope);
  }

  /// Resolved a symbol table entry to a value.  Emission of error is optional.
  ParseResult resolveSymbolEntry(Value &result, SymbolValueEntry &entry,
                                 SMLoc loc, bool fatal = true);

  /// Resolved a symbol table entry if it is an expanded bundle e.g. from an
  /// instance.  Emission of error is optional.
  ParseResult resolveSymbolEntry(Value &result, SymbolValueEntry &entry,
                                 StringRef field, SMLoc loc);

  /// Look up the specified name, emitting an error and returning failure if the
  /// name is unknown.
  ParseResult lookupSymbolEntry(SymbolValueEntry &result, StringRef name,
                                SMLoc loc);

  UnbundledValueEntry &getUnbundledEntry(unsigned index) {
    assert(index < unbundledValues.size());
    return unbundledValues[index];
  }

  /// This contains one entry for each value in FIRRTL that is represented as a
  /// bundle type in the FIRRTL spec but for which we represent as an exploded
  /// set of elements in the FIRRTL dialect.
  UnbundledValuesList unbundledValues;

  /// Provide a symbol table scope that automatically pops all the entries off
  /// the symbol table when the scope is exited.
  struct ContextScope {
    friend struct FIRModuleContext;
    ContextScope(FIRModuleContext &moduleContext, Block *block)
        : moduleContext(moduleContext), block(block),
          previousScope(moduleContext.currentScope) {
      moduleContext.currentScope = this;
      moduleContext.scopeMap[block] = this;
    }
    ~ContextScope() {
      // Mark all entries in this scope as being invalid.  We track validity
      // through the SMLoc field instead of deleting entries.
      for (auto *entryPtr : scopedDecls)
        entryPtr->second.first = SMLoc();
      // Erase the scoped subacceses from the cache. If the block is deleted we
      // could resuse the memory, although the chances are quite small.
      for (auto subaccess : scopedSubaccesses)
        moduleContext.subaccessCache.erase(subaccess);
      // Erase this context from the map.
      moduleContext.scopeMap.erase(block);
      // Reset to the previous scope.
      moduleContext.currentScope = previousScope;
    }

  private:
    void operator=(const ContextScope &) = delete;
    ContextScope(const ContextScope &) = delete;

    FIRModuleContext &moduleContext;
    Block *block;
    ContextScope *previousScope;
    std::vector<ModuleSymbolTableEntry *> scopedDecls;
    std::vector<std::pair<Value, unsigned>> scopedSubaccesses;
  };

private:
  /// This symbol table holds the names of ports, wires, and other local decls.
  /// This is scoped because conditional statements introduce subscopes.
  ModuleSymbolTable symbolTable;

  /// This is a cache of subindex and subfield operations so we don't constantly
  /// recreate large chains of them.  This maps a bundle value + index to the
  /// subaccess result.
  SubaccessCache subaccessCache;

  /// This maps a block to related ContextScope.
  DenseMap<Block *, ContextScope *> scopeMap;

  /// If non-null, all new entries added to the symbol table are added to this
  /// list.  This allows us to "pop" the entries by resetting them to null when
  /// scope is exited.
  ContextScope *currentScope = nullptr;
};

} // end anonymous namespace

/// Add a symbol entry with the specified name, returning failure if the name
/// is already defined.
///
/// When 'insertNameIntoGlobalScope' is true, we don't allow the name to be
/// popped.  This is a workaround for (firrtl scala bug) that should eventually
/// be fixed.
ParseResult FIRModuleContext::addSymbolEntry(StringRef name,
                                             SymbolValueEntry entry, SMLoc loc,
                                             bool insertNameIntoGlobalScope) {
  // Do a lookup by trying to do an insertion.  Do so in a way that we can tell
  // if we hit a missing element (SMLoc is null).
  auto entryIt =
      symbolTable.try_emplace(name, SMLoc(), SymbolValueEntry()).first;
  if (entryIt->second.first.isValid()) {
    emitError(loc, "redefinition of name '" + name + "'")
            .attachNote(translateLocation(entryIt->second.first))
        << "previous definition here";
    return failure();
  }

  // If we didn't have a hit, then record the location, and remember that this
  // was new to this scope.
  entryIt->second = {loc, entry};
  if (currentScope && !insertNameIntoGlobalScope)
    currentScope->scopedDecls.push_back(&*entryIt);

  return success();
}

/// Look up the specified name, emitting an error and returning null if the
/// name is unknown.
ParseResult FIRModuleContext::lookupSymbolEntry(SymbolValueEntry &result,
                                                StringRef name, SMLoc loc) {
  auto &entry = symbolTable[name];
  if (!entry.first.isValid())
    return emitError(loc, "use of unknown declaration '" + name + "'");
  result = entry.second;
  assert(result && "name in symbol table without definition");
  return success();
}

ParseResult FIRModuleContext::resolveSymbolEntry(Value &result,
                                                 SymbolValueEntry &entry,
                                                 SMLoc loc, bool fatal) {
  if (!entry.is<Value>()) {
    if (fatal)
      emitError(loc, "bundle value should only be used from subfield");
    return failure();
  }
  result = entry.get<Value>();
  return success();
}

ParseResult FIRModuleContext::resolveSymbolEntry(Value &result,
                                                 SymbolValueEntry &entry,
                                                 StringRef fieldName,
                                                 SMLoc loc) {
  if (!entry.is<UnbundledID>()) {
    emitError(loc, "value should not be used from subfield");
    return failure();
  }

  auto fieldAttr = StringAttr::get(getContext(), fieldName);

  unsigned unbundledId = entry.get<UnbundledID>() - 1;
  assert(unbundledId < unbundledValues.size());
  UnbundledValueEntry &ubEntry = unbundledValues[unbundledId];
  for (auto elt : ubEntry) {
    if (elt.first == fieldAttr) {
      result = elt.second;
      break;
    }
  }
  if (!result) {
    emitError(loc, "use of invalid field name '")
        << fieldName << "' on bundle value";
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// FIRStmtParser
//===----------------------------------------------------------------------===//

namespace {
/// This class is used when building expression nodes for a statement: we need
/// to parse a bunch of expressions and build MLIR operations for them, and then
/// we see the locator that specifies the location for those operations
/// afterward.
///
/// It is wasteful in time and memory to create a bunch of temporary FileLineCol
/// location's that point into the .fir file when they're destined to get
/// overwritten by a location specified by a Locator.  To avoid this, we create
/// all of the operations with a temporary location on them, then remember the
/// [Operation*, SMLoc] pair for the newly created operation.
///
/// At the end of the operation we'll see a Locator (or not).  If we see a
/// locator, we apply it to all the operations we've parsed and we're done.  If
/// not, we lazily create the locators in the .fir file.
struct LazyLocationListener : public OpBuilder::Listener {
  LazyLocationListener(OpBuilder &builder) : builder(builder) {
    assert(builder.getListener() == nullptr);
    builder.setListener(this);
  }

  ~LazyLocationListener() {
    assert(subOps.empty() && "didn't process parsed operations");
    assert(builder.getListener() == this);
    builder.setListener(nullptr);
  }

  void startStatement() {
    assert(!isActive && "Already processing a statement");
    isActive = true;
  }

  /// This is called when done with each statement.  This applies the locations
  /// to each statement.
  void endStatement(FIRParser &parser) {
    assert(isActive && "Not parsing a statement");

    // If we have a symbolic location, apply it to any subOps specified.
    if (infoLoc) {
      for (auto opAndSMLoc : subOps) {
        // Follow user preference to either only use @info locations,
        // or apply a fused location with @info and file loc.
        using ILH = FIRParserOptions::InfoLocHandling;
        switch (parser.getConstants().options.infoLocatorHandling) {
        case ILH::IgnoreInfo:
          // Shouldn't have an infoLoc, but if we do ignore it.
          opAndSMLoc.first->setLoc(parser.translateLocation(opAndSMLoc.second));
          break;
        case ILH::PreferInfo:
          opAndSMLoc.first->setLoc(infoLoc);
          break;
        case ILH::FusedInfo:
          opAndSMLoc.first->setLoc(FusedLoc::get(
              infoLoc.getContext(),
              {infoLoc, parser.translateLocation(opAndSMLoc.second)}));
          break;
        }
      }
    } else {
      // If we don't, translate all the individual SMLoc's to Location objects
      // in the .fir file.
      for (auto opAndSMLoc : subOps)
        opAndSMLoc.first->setLoc(parser.translateLocation(opAndSMLoc.second));
    }

    // Reset our state.
    isActive = false;
    infoLoc = LocationAttr();
    currentSMLoc = SMLoc();
    subOps.clear();
  }

  /// Specify the location to be used for the next operations that are created.
  void setLoc(SMLoc loc) { currentSMLoc = loc; }

  /// When a @Info locator is parsed, this method captures it.
  void setInfoLoc(LocationAttr loc) {
    assert(!infoLoc && "Info location multiply specified");
    infoLoc = loc;
  }

  // Notification handler for when an operation is inserted into the builder.
  /// `op` is the operation that was inserted.
  void notifyOperationInserted(Operation *op) override {
    assert(currentSMLoc != SMLoc() && "No .fir file location specified");
    assert(isActive && "Not parsing a statement");
    subOps.push_back({op, currentSMLoc});
  }

private:
  /// This is set to true while parsing a statement.  It is used for assertions.
  bool isActive = false;

  /// This is the current position in the source file that the next operation
  /// will be parsed into.
  SMLoc currentSMLoc;

  /// This is the @ location attribute for the current statement, or null if
  /// not set.
  LocationAttr infoLoc;

  /// This is the builder we're installed into.
  OpBuilder &builder;

  /// This is the set of operations we've enqueued along with their location in
  /// the source file.
  SmallVector<std::pair<Operation *, SMLoc>, 8> subOps;

  void operator=(const LazyLocationListener &) = delete;
  LazyLocationListener(const LazyLocationListener &) = delete;
};
} // end anonymous namespace

namespace {
/// This class implements logic and state for parsing statements, suites, and
/// similar module body constructs.
struct FIRStmtParser : public FIRParser {
  explicit FIRStmtParser(Block &blockToInsertInto,
                         FIRModuleContext &moduleContext,
                         ModuleNamespace &modNameSpace, FIRVersion &version,
                         SymbolRefAttr groupSym = {})
      : FIRParser(moduleContext.getConstants(), moduleContext.getLexer(),
                  version),
        builder(UnknownLoc::get(getContext()), getContext()),
        locationProcessor(this->builder), moduleContext(moduleContext),
        modNameSpace(modNameSpace), groupSym(groupSym) {
    builder.setInsertionPointToEnd(&blockToInsertInto);
  }

  ParseResult parseSimpleStmt(unsigned stmtIndent);
  ParseResult parseSimpleStmtBlock(unsigned indent);

private:
  ParseResult parseSimpleStmtImpl(unsigned stmtIndent);

  /// Attach invalid values to every element of the value.
  void emitInvalidate(Value val, Flow flow);

  // The FIRRTL specification describes Invalidates as a statement with
  // implicit connect semantics.  The FIRRTL dialect models it as a primitive
  // that returns an "Invalid Value", followed by an explicit connect to make
  // the representation simpler and more consistent.
  void emitInvalidate(Value val) { emitInvalidate(val, foldFlow(val)); }

  /// Emit the logic for a partial connect using standard connect.
  void emitPartialConnect(ImplicitLocOpBuilder &builder, Value dst, Value src);

  /// Parse an @info marker if present and inform locationProcessor about it.
  ParseResult parseOptionalInfo() {
    LocationAttr loc;
    if (failed(parseOptionalInfoLocator(loc)))
      return failure();
    locationProcessor.setInfoLoc(loc);
    return success();
  }

  // Exp Parsing
  ParseResult parseExpImpl(Value &result, const Twine &message,
                           bool isLeadingStmt);
  ParseResult parseExp(Value &result, const Twine &message) {
    return parseExpImpl(result, message, /*isLeadingStmt:*/ false);
  }
  ParseResult parseExpLeadingStmt(Value &result, const Twine &message) {
    return parseExpImpl(result, message, /*isLeadingStmt:*/ true);
  }
  ParseResult parseEnumExp(Value &result);
  ParseResult parseRefExp(Value &result, const Twine &message);
  ParseResult parseStaticRefExp(Value &result, const Twine &message);

  template <typename subop>
  FailureOr<Value> emitCachedSubAccess(Value base,
                                       ArrayRef<NamedAttribute> attrs,
                                       unsigned indexNo, SMLoc loc);
  ParseResult parseOptionalExpPostscript(Value &result,
                                         bool allowDynamic = true);
  ParseResult parsePostFixFieldId(Value &result);
  ParseResult parsePostFixIntSubscript(Value &result);
  ParseResult parsePostFixDynamicSubscript(Value &result);
  ParseResult parsePrimExp(Value &result);
  ParseResult parseIntegerLiteralExp(Value &result);

  std::optional<ParseResult> parseExpWithLeadingKeyword(FIRToken keyword);

  // Stmt Parsing
  ParseResult parseSubBlock(Block &blockToInsertInto, unsigned indent,
                            SymbolRefAttr groupSym);
  ParseResult parseAttach();
  ParseResult parseMemPort(MemDirAttr direction);
  ParseResult parsePrintf();
  ParseResult parseSkip();
  ParseResult parseStop();
  ParseResult parseAssert();
  ParseResult parseAssume();
  ParseResult parseCover();
  ParseResult parseWhen(unsigned whenIndent);
  ParseResult parseMatch(unsigned matchIndent);
  ParseResult parseRefDefine();
  ParseResult parseRefForce();
  ParseResult parseRefForceInitial();
  ParseResult parseRefRelease();
  ParseResult parseRefReleaseInitial();
  ParseResult parseRefRead(Value &result);
  ParseResult parseProbe(Value &result);
  ParseResult parsePropAssign();
  ParseResult parseRWProbe(Value &result);
  ParseResult parseLeadingExpStmt(Value lhs);
  ParseResult parseConnect();
  ParseResult parseInvalidate();
  ParseResult parseGroup(unsigned indent);

  // Declarations
  ParseResult parseInstance();
  ParseResult parseCombMem();
  ParseResult parseSeqMem();
  ParseResult parseMem(unsigned memIndent);
  ParseResult parseNode();
  ParseResult parseWire();
  ParseResult parseRegister(unsigned regIndent);
  ParseResult parseRegisterWithReset();

  // The builder to build into.
  ImplicitLocOpBuilder builder;
  LazyLocationListener locationProcessor;

  // Extra information maintained across a module.
  FIRModuleContext &moduleContext;

  ModuleNamespace &modNameSpace;

  // An optional symbol that contains the current group that we are in.  This is
  // used to construct a nested symbol for a group definition operation.
  SymbolRefAttr groupSym;
};

} // end anonymous namespace

/// Attach invalid values to every element of the value.
// NOLINTNEXTLINE(misc-no-recursion)
void FIRStmtParser::emitInvalidate(Value val, Flow flow) {
  auto tpe = type_dyn_cast<FIRRTLBaseType>(val.getType());
  // Invalidate does nothing for non-base types.
  // When aggregates-of-refs are supported, instead check 'containsReference'
  // below.
  if (!tpe)
    return;

  auto props = tpe.getRecursiveTypeProperties();
  if (props.isPassive && !props.containsAnalog) {
    if (flow == Flow::Source)
      return;
    emitConnect(builder, val, builder.create<InvalidValueOp>(tpe));
    return;
  }

  // Recurse until we hit passive leaves.  Connect any leaves which have sink or
  // duplex flow.
  //
  // TODO: This is very similar to connect expansion in the LowerTypes pass
  // works.  Find a way to unify this with methods common to LowerTypes or to
  // have LowerTypes to the actual work here, e.g., emitting a partial connect
  // to only the leaf sources.
  TypeSwitch<FIRRTLType>(tpe)
      .Case<BundleType>([&](auto tpe) {
        for (size_t i = 0, e = tpe.getNumElements(); i < e; ++i) {
          auto &subfield = moduleContext.getCachedSubaccess(val, i);
          if (!subfield) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfterValue(val);
            subfield = builder.create<SubfieldOp>(val, i);
          }
          emitInvalidate(subfield,
                         tpe.getElement(i).isFlip ? swapFlow(flow) : flow);
        }
      })
      .Case<FVectorType>([&](auto tpe) {
        auto tpex = tpe.getElementType();
        for (size_t i = 0, e = tpe.getNumElements(); i != e; ++i) {
          auto &subindex = moduleContext.getCachedSubaccess(val, i);
          if (!subindex) {
            OpBuilder::InsertionGuard guard(builder);
            builder.setInsertionPointAfterValue(val);
            subindex = builder.create<SubindexOp>(tpex, val, i);
          }
          emitInvalidate(subindex, flow);
        }
      });
}

void FIRStmtParser::emitPartialConnect(ImplicitLocOpBuilder &builder, Value dst,
                                       Value src) {
  auto dstType = type_dyn_cast<FIRRTLBaseType>(dst.getType());
  auto srcType = type_dyn_cast<FIRRTLBaseType>(src.getType());
  if (!dstType || !srcType)
    return emitConnect(builder, dst, src);

  if (type_isa<AnalogType>(dstType)) {
    builder.create<AttachOp>(ArrayRef<Value>{dst, src});
  } else if (dstType == srcType && !dstType.containsAnalog()) {
    emitConnect(builder, dst, src);
  } else if (auto dstBundle = type_dyn_cast<BundleType>(dstType)) {
    auto srcBundle = type_cast<BundleType>(srcType);
    auto numElements = dstBundle.getNumElements();
    for (size_t dstIndex = 0; dstIndex < numElements; ++dstIndex) {
      // Find a matching field by name in the other bundle.
      auto &dstElement = dstBundle.getElements()[dstIndex];
      auto name = dstElement.name;
      auto maybe = srcBundle.getElementIndex(name);
      // If there was no matching field name, don't connect this one.
      if (!maybe)
        continue;
      auto dstRef = moduleContext.getCachedSubaccess(dst, dstIndex);
      if (!dstRef) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfterValue(dst);
        dstRef = builder.create<SubfieldOp>(dst, dstIndex);
      }
      // We are pulling two fields from the cache. If the dstField was a
      // pointer into the cache, then the lookup for srcField might invalidate
      // it. So, we just copy dstField into a local.
      auto dstField = dstRef;
      auto srcIndex = *maybe;
      auto &srcField = moduleContext.getCachedSubaccess(src, srcIndex);
      if (!srcField) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfterValue(src);
        srcField = builder.create<SubfieldOp>(src, srcIndex);
      }
      if (!dstElement.isFlip)
        emitPartialConnect(builder, dstField, srcField);
      else
        emitPartialConnect(builder, srcField, dstField);
    }
  } else if (auto dstVector = type_dyn_cast<FVectorType>(dstType)) {
    auto srcVector = type_cast<FVectorType>(srcType);
    auto dstNumElements = dstVector.getNumElements();
    auto srcNumEelemnts = srcVector.getNumElements();
    // Partial connect will connect all elements up to the end of the array.
    auto numElements = std::min(dstNumElements, srcNumEelemnts);
    for (size_t i = 0; i != numElements; ++i) {
      auto &dstRef = moduleContext.getCachedSubaccess(dst, i);
      if (!dstRef) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfterValue(dst);
        dstRef = builder.create<SubindexOp>(dst, i);
      }
      auto dstField = dstRef; // copy to ensure not invalidated
      auto &srcField = moduleContext.getCachedSubaccess(src, i);
      if (!srcField) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfterValue(src);
        srcField = builder.create<SubindexOp>(src, i);
      }
      emitPartialConnect(builder, dstField, srcField);
    }
  } else {
    emitConnect(builder, dst, src);
  }
}

//===-------------------------------
// FIRStmtParser Expression Parsing.

/// Parse the 'exp' grammar, returning all of the suboperations in the
/// specified vector, and the ultimate SSA value in value.
///
///  exp ::= id    // Ref
///      ::= prim
///      ::= integer-literal-exp
///      ::= enum-exp
///      ::= 'String(' stringLit ')'
///      ::= exp '.' fieldId
///      ::= exp '[' intLit ']'
/// XX   ::= exp '.' DoubleLit // TODO Workaround for #470
///      ::= exp '[' exp ']'
///
///
/// If 'isLeadingStmt' is true, then this is being called to parse the first
/// expression in a statement.  We can handle some weird cases due to this if
/// we end up parsing the whole statement.  In that case we return success, but
/// set the 'result' value to null.
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRStmtParser::parseExpImpl(Value &result, const Twine &message,
                                        bool isLeadingStmt) {
  switch (getToken().getKind()) {

    // Handle all primitive's.
#define TOK_LPKEYWORD_PRIM(SPELLING, CLASS, NUMOPERANDS)                       \
  case FIRToken::lp_##SPELLING:
#include "FIRTokenKinds.def"
    if (parsePrimExp(result))
      return failure();
    break;

  case FIRToken::l_brace_bar:
    if (isLeadingStmt)
      return emitError("unexpected enumeration as start of statement");
    if (parseEnumExp(result))
      return failure();
    break;
  case FIRToken::lp_read:
    if (isLeadingStmt)
      return emitError("unexpected read() as start of statement");
    if (parseRefRead(result))
      return failure();
    break;
  case FIRToken::lp_probe:
    if (isLeadingStmt)
      return emitError("unexpected probe() as start of statement");
    if (parseProbe(result))
      return failure();
    break;
  case FIRToken::lp_rwprobe:
    if (isLeadingStmt)
      return emitError("unexpected rwprobe() as start of statement");
    if (parseRWProbe(result))
      return failure();
    break;

  case FIRToken::kw_UInt:
  case FIRToken::kw_SInt:
    if (parseIntegerLiteralExp(result))
      return failure();
    break;
  case FIRToken::kw_String: {
    if (FIRVersion::compare(version, FIRVersion({3, 1, 0})) < 0)
      return emitError() << "unexpected token: Properties are a FIRRTL 3.1.0+ "
                            "feature, but the specified FIRRTL version was "
                         << version;
    locationProcessor.setLoc(getToken().getLoc());
    consumeToken(FIRToken::kw_String);
    StringRef spelling;
    if (parseToken(FIRToken::l_paren, "expected '(' in String expression") ||
        parseGetSpelling(spelling) ||
        parseToken(FIRToken::string,
                   "expected string literal in String expression") ||
        parseToken(FIRToken::r_paren, "expected ')' in String expression"))
      return failure();
    result = builder.create<StringConstantOp>(
        builder.getStringAttr(FIRToken::getStringValue(spelling)));
    break;
  }
  case FIRToken::kw_Integer: {
    if (FIRVersion::compare(version, FIRVersion({3, 1, 0})) < 0)
      return emitError() << "unexpected token: Integers are a FIRRTL 3.1.0+ "
                            "feature, but the specified FIRRTL version was "
                         << version;

    locationProcessor.setLoc(getToken().getLoc());
    consumeToken(FIRToken::kw_Integer);
    APInt value;
    if (parseToken(FIRToken::l_paren, "expected '(' in Integer expression") ||
        parseIntLit(value, "expected integer literal in Integer expression") ||
        parseToken(FIRToken::r_paren, "expected ')' in Integer expression"))
      return failure();
    result =
        builder.create<FIntegerConstantOp>(APSInt(value, /*isUnsigned=*/false));
    break;
  }

    // Otherwise there are a bunch of keywords that are treated as identifiers
    // try them.
  case FIRToken::identifier: // exp ::= id
  case FIRToken::literal_identifier:
  default: {
    StringRef name;
    auto loc = getToken().getLoc();
    SymbolValueEntry symtabEntry;
    if (parseId(name, message) ||
        moduleContext.lookupSymbolEntry(symtabEntry, name, loc))
      return failure();

    // If we looked up a normal value, then we're done.
    if (!moduleContext.resolveSymbolEntry(result, symtabEntry, loc, false))
      break;

    assert(symtabEntry.is<UnbundledID>() && "should be an instance");

    // Otherwise we referred to an implicitly bundled value.  We *must* be in
    // the midst of processing a field ID reference or 'is invalid'.  If not,
    // this is an error.
    if (isLeadingStmt && consumeIf(FIRToken::kw_is)) {
      if (parseToken(FIRToken::kw_invalid, "expected 'invalid'") ||
          parseOptionalInfo())
        return failure();

      locationProcessor.setLoc(loc);
      // Invalidate all of the results of the bundled value.
      unsigned unbundledId = symtabEntry.get<UnbundledID>() - 1;
      UnbundledValueEntry &ubEntry =
          moduleContext.getUnbundledEntry(unbundledId);
      for (auto elt : ubEntry)
        emitInvalidate(elt.second);

      // Signify that we parsed the whole statement.
      result = Value();
      return success();
    }

    // Handle the normal "instance.x" reference.
    StringRef fieldName;
    if (parseToken(FIRToken::period, "expected '.' in field reference") ||
        parseFieldId(fieldName, "expected field name") ||
        moduleContext.resolveSymbolEntry(result, symtabEntry, fieldName, loc))
      return failure();
    break;
  }
  }

  return parseOptionalExpPostscript(result);
}

/// Parse the postfix productions of expression after the leading expression
/// has been parsed.
///
///  exp ::= exp '.' fieldId
///      ::= exp '[' intLit ']'
/// XX   ::= exp '.' DoubleLit // TODO Workaround for #470
///      ::= exp '[' exp ']'
ParseResult FIRStmtParser::parseOptionalExpPostscript(Value &result,
                                                      bool allowDynamic) {

  // Handle postfix expressions.
  while (true) {
    // Subfield: exp ::= exp '.' fieldId
    if (consumeIf(FIRToken::period)) {
      if (parsePostFixFieldId(result))
        return failure();

      continue;
    }

    // Subindex: exp ::= exp '[' intLit ']' | exp '[' exp ']'
    if (consumeIf(FIRToken::l_square)) {
      if (getToken().isAny(FIRToken::integer, FIRToken::string)) {
        if (parsePostFixIntSubscript(result))
          return failure();
        continue;
      }
      if (!allowDynamic)
        return emitError("subaccess not allowed here");
      if (parsePostFixDynamicSubscript(result))
        return failure();

      continue;
    }

    return success();
  }
}

template <typename subop>
FailureOr<Value>
FIRStmtParser::emitCachedSubAccess(Value base, ArrayRef<NamedAttribute> attrs,
                                   unsigned indexNo, SMLoc loc) {
  // Make sure the field name matches up with the input value's type and
  // compute the result type for the expression.
  auto resultType = subop::inferReturnType({base}, attrs, {});
  if (!resultType) {
    // Emit the error at the right location.  translateLocation is expensive.
    (void)subop::inferReturnType({base}, attrs, translateLocation(loc));
    return failure();
  }

  // Check if we already have created this Subindex op.
  auto &value = moduleContext.getCachedSubaccess(base, indexNo);
  if (value)
    return value;

  // Create the result operation, inserting at the location of the declaration.
  // This will cache the subfield operation for further uses.
  locationProcessor.setLoc(loc);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfterValue(base);
  auto op = builder.create<subop>(resultType, base, attrs);

  // Insert the newly created operation into the cache.
  return value = op.getResult();
}

/// exp ::= exp '.' fieldId
///
/// The "exp '.'" part of the production has already been parsed.
///
ParseResult FIRStmtParser::parsePostFixFieldId(Value &result) {
  auto loc = getToken().getLoc();
  SmallVector<StringRef, 3> fields;
  if (parseFieldIdSeq(fields, "expected field name"))
    return failure();
  for (auto fieldName : fields) {
    std::optional<unsigned> indexV;
    auto type = result.getType();
    if (auto refTy = type_dyn_cast<RefType>(type))
      type = refTy.getType();
    if (auto bundle = type_dyn_cast<BundleType>(type))
      indexV = bundle.getElementIndex(fieldName);
    else if (auto bundle = type_dyn_cast<OpenBundleType>(type))
      indexV = bundle.getElementIndex(fieldName);
    else
      return emitError(loc, "subfield requires bundle operand ");
    if (!indexV)
      return emitError(loc, "unknown field '" + fieldName + "' in bundle type ")
             << result.getType();
    auto indexNo = *indexV;

    FailureOr<Value> subResult;
    if (type_isa<RefType>(result.getType())) {
      NamedAttribute attrs = {getConstants().indexIdentifier,
                              builder.getI32IntegerAttr(indexNo)};
      subResult = emitCachedSubAccess<RefSubOp>(result, attrs, indexNo, loc);
    } else {
      NamedAttribute attrs = {getConstants().fieldIndexIdentifier,
                              builder.getI32IntegerAttr(indexNo)};
      if (type_isa<BundleType>(type))
        subResult =
            emitCachedSubAccess<SubfieldOp>(result, attrs, indexNo, loc);
      else
        subResult =
            emitCachedSubAccess<OpenSubfieldOp>(result, attrs, indexNo, loc);
    }

    if (failed(subResult))
      return failure();
    result = *subResult;
  }
  return success();
}

/// exp ::= exp '[' intLit ']'
///
/// The "exp '['" part of the production has already been parsed.
///
ParseResult FIRStmtParser::parsePostFixIntSubscript(Value &result) {
  auto loc = getToken().getLoc();
  int32_t indexNo;
  if (parseIntLit(indexNo, "expected index") ||
      parseToken(FIRToken::r_square, "expected ']'"))
    return failure();

  if (indexNo < 0)
    return emitError(loc, "invalid index specifier"), failure();

  // Make sure the index expression is valid and compute the result type for the
  // expression.
  // TODO: This should ideally be folded into a `tryCreate` method on the
  // builder (https://llvm.discourse.group/t/3504).
  NamedAttribute attrs = {getConstants().indexIdentifier,
                          builder.getI32IntegerAttr(indexNo)};

  FailureOr<Value> subResult;
  if (type_isa<RefType>(result.getType()))
    subResult = emitCachedSubAccess<RefSubOp>(result, attrs, indexNo, loc);
  else if (type_isa<FVectorType>(result.getType()))
    subResult = emitCachedSubAccess<SubindexOp>(result, attrs, indexNo, loc);
  else
    subResult =
        emitCachedSubAccess<OpenSubindexOp>(result, attrs, indexNo, loc);

  if (failed(subResult))
    return failure();
  result = *subResult;
  return success();
}

/// exp ::= exp '[' exp ']'
///
/// The "exp '['" part of the production has already been parsed.
///
ParseResult FIRStmtParser::parsePostFixDynamicSubscript(Value &result) {
  auto loc = getToken().getLoc();
  Value index;
  if (parseExp(index, "expected subscript index expression") ||
      parseToken(FIRToken::r_square, "expected ']' in subscript"))
    return failure();

  // If the index expression is a flip type, strip it off.
  auto indexType = type_dyn_cast<FIRRTLBaseType>(index.getType());
  if (!indexType)
    return emitError("expected base type for index expression");
  indexType = indexType.getPassiveType();
  locationProcessor.setLoc(loc);

  // Make sure the index expression is valid and compute the result type for the
  // expression.
  auto resultType = SubaccessOp::inferReturnType({result, index}, {}, {});
  if (!resultType) {
    // Emit the error at the right location.  translateLocation is expensive.
    (void)SubaccessOp::inferReturnType({result, index}, {},
                                       translateLocation(loc));
    return failure();
  }

  // Create the result operation.
  auto op = builder.create<SubaccessOp>(resultType, result, index);
  result = op.getResult();
  return success();
}

/// prim ::= primop exp* intLit*  ')'
ParseResult FIRStmtParser::parsePrimExp(Value &result) {
  auto kind = getToken().getKind();
  auto loc = getToken().getLoc();
  consumeToken();

  // Parse the operands and constant integer arguments.
  SmallVector<Value, 3> operands;
  SmallVector<int64_t, 3> integers;
  if (parseListUntil(FIRToken::r_paren, [&]() -> ParseResult {
        // Handle the integer constant case if present.
        if (getToken().isAny(FIRToken::integer, FIRToken::signed_integer,
                             FIRToken::string)) {
          integers.push_back(0);
          return parseIntLit(integers.back(), "expected integer");
        }

        // Otherwise it must be a value operand.  These must all come before the
        // integers.
        if (!integers.empty())
          return emitError("expected more integer constants"), failure();

        Value operand;
        if (parseExp(operand, "expected expression in primitive operand"))
          return failure();

        locationProcessor.setLoc(loc);

        operands.push_back(operand);
        return success();
      }))
    return failure();

  locationProcessor.setLoc(loc);

  SmallVector<FIRRTLType, 3> opTypes;
  for (auto v : operands)
    opTypes.push_back(type_cast<FIRRTLType>(v.getType()));

  unsigned numOperandsExpected;
  SmallVector<StringAttr, 2> attrNames;

  // Get information about the primitive in question.
  switch (kind) {
  default:
    emitError(loc, "primitive not supported yet");
    return failure();
#define TOK_LPKEYWORD_PRIM(SPELLING, CLASS, NUMOPERANDS)                       \
  case FIRToken::lp_##SPELLING:                                                \
    numOperandsExpected = NUMOPERANDS;                                         \
    break;
#include "FIRTokenKinds.def"
  }
  // Don't add code here, we want these two switch statements to be fused by
  // the compiler.
  switch (kind) {
  default:
    break;
  case FIRToken::lp_bits:
    attrNames.push_back(getConstants().hiIdentifier); // "hi"
    attrNames.push_back(getConstants().loIdentifier); // "lo"
    break;
  case FIRToken::lp_head:
  case FIRToken::lp_pad:
  case FIRToken::lp_shl:
  case FIRToken::lp_shr:
  case FIRToken::lp_tail:
    attrNames.push_back(getConstants().amountIdentifier);
    break;
  }

  if (operands.size() != numOperandsExpected) {
    assert(numOperandsExpected <= 3);
    static const char *numberName[] = {"zero", "one", "two", "three"};
    const char *optionalS = &"s"[numOperandsExpected == 1];
    return emitError(loc, "operation requires ")
           << numberName[numOperandsExpected] << " operand" << optionalS;
  }

  if (integers.size() != attrNames.size()) {
    emitError(loc, "expected ") << attrNames.size() << " constant arguments";
    return failure();
  }

  NamedAttrList attrs;
  for (size_t i = 0, e = attrNames.size(); i != e; ++i)
    attrs.append(attrNames[i], builder.getI32IntegerAttr(integers[i]));

  switch (kind) {
  default:
    emitError(loc, "primitive not supported yet");
    return failure();

#define TOK_LPKEYWORD_PRIM(SPELLING, CLASS, NUMOPERANDS)                       \
  case FIRToken::lp_##SPELLING: {                                              \
    auto resultTy = CLASS::inferReturnType(operands, attrs, {});               \
    if (!resultTy) {                                                           \
      /* only call translateLocation on an error case, it is expensive. */     \
      (void)CLASS::validateAndInferReturnType(operands, attrs,                 \
                                              translateLocation(loc));         \
      return failure();                                                        \
    }                                                                          \
    result = builder.create<CLASS>(resultTy, operands, attrs);                 \
    return success();                                                          \
  }
#include "FIRTokenKinds.def"
  }

  llvm_unreachable("all cases should return");
}

/// integer-literal-exp ::= 'UInt' optional-width '(' intLit ')'
///                     ::= 'SInt' optional-width '(' intLit ')'
ParseResult FIRStmtParser::parseIntegerLiteralExp(Value &result) {
  bool isSigned = getToken().is(FIRToken::kw_SInt);
  auto loc = getToken().getLoc();
  consumeToken();

  // Parse a width specifier if present.
  int32_t width;
  APInt value;
  if (parseOptionalWidth(width) ||
      parseToken(FIRToken::l_paren, "expected '(' in integer expression") ||
      parseIntLit(value, "expected integer value") ||
      parseToken(FIRToken::r_paren, "expected ')' in integer expression"))
    return failure();

  // Construct an integer attribute of the right width.
  // Literals are parsed as 'const' types.
  auto type = IntType::get(builder.getContext(), isSigned, width, true);

  IntegerType::SignednessSemantics signedness =
      isSigned ? IntegerType::Signed : IntegerType::Unsigned;
  if (width == 0) {
    if (!value.isZero())
      return emitError(loc, "zero bit constant must be zero");
    value = value.trunc(0);
  } else if (width != -1) {
    // Convert to the type's width, checking value fits in destination width.
    bool valueFits = isSigned ? value.isSignedIntN(width) : value.isIntN(width);
    if (!valueFits)
      return emitError(loc, "initializer too wide for declared width");
    value = isSigned ? value.sextOrTrunc(width) : value.zextOrTrunc(width);
  }

  Type attrType =
      IntegerType::get(type.getContext(), value.getBitWidth(), signedness);
  auto attr = builder.getIntegerAttr(attrType, value);

  // Check to see if we've already created this constant.  If so, reuse it.
  auto &entry = moduleContext.constantCache[{attr, type}];
  if (entry) {
    // If we already had an entry, reuse it.
    result = entry;
    return success();
  }

  locationProcessor.setLoc(loc);
  result = moduleContext.getCachedConstantInt(builder, attr, type, value);
  return success();
}

/// The .fir grammar has the annoying property where:
/// 1) some statements start with keywords
/// 2) some start with an expression
/// 3) it allows the 'reference' expression to either be an identifier or a
///    keyword.
///
/// One example of this is something like, where this is not a register decl:
///   reg <- thing
///
/// Solving this requires lookahead to the second token.  We handle it by
///  factoring the lookahead inline into the code to keep the parser fast.
///
/// As such, statements that start with a leading keyword call this method to
/// check to see if the keyword they consumed was actually the start of an
/// expression.  If so, they parse the expression-based statement and return the
/// parser result.  If not, they return None and the statement is parsed like
/// normal.
std::optional<ParseResult>
FIRStmtParser::parseExpWithLeadingKeyword(FIRToken keyword) {
  switch (getToken().getKind()) {
  default:
    // This isn't part of an expression, and isn't part of a statement.
    return std::nullopt;

  case FIRToken::period:     // exp `.` identifier
  case FIRToken::l_square:   // exp `[` index `]`
  case FIRToken::kw_is:      // exp is invalid
  case FIRToken::less_equal: // exp <= thing
  case FIRToken::less_minus: // exp <- thing
    break;
  }

  Value lhs;
  SymbolValueEntry symtabEntry;
  auto loc = keyword.getLoc();

  if (moduleContext.lookupSymbolEntry(symtabEntry, keyword.getSpelling(), loc))
    return ParseResult(failure());

  // If we have a '.', we might have a symbol or an expanded port.  If we
  // resolve to a symbol, use that, otherwise check for expanded bundles of
  // other ops.
  // Non '.' ops take the plain symbol path.
  if (moduleContext.resolveSymbolEntry(lhs, symtabEntry, loc, false)) {
    // Ok if the base name didn't resolve by itself, it might be part of an
    // expanded dot reference.  That doesn't work then we fail.
    if (!consumeIf(FIRToken::period))
      return ParseResult(failure());

    StringRef fieldName;
    if (parseFieldId(fieldName, "expected field name") ||
        moduleContext.resolveSymbolEntry(lhs, symtabEntry, fieldName, loc))
      return ParseResult(failure());
  }

  // Parse any further trailing things like "mem.x.y".
  if (parseOptionalExpPostscript(lhs))
    return ParseResult(failure());

  return parseLeadingExpStmt(lhs);
}
//===-----------------------------
// FIRStmtParser Statement Parsing

/// simple_stmt_block ::= simple_stmt*
ParseResult FIRStmtParser::parseSimpleStmtBlock(unsigned indent) {
  while (true) {
    // The outer level parser can handle these tokens.
    if (getToken().isAny(FIRToken::eof, FIRToken::error))
      return success();

    auto subIndent = getIndentation();
    if (!subIndent.has_value())
      return emitError("expected statement to be on its own line"), failure();

    if (*subIndent <= indent)
      return success();

    // Let the statement parser handle this.
    if (parseSimpleStmt(*subIndent))
      return failure();
  }
}

ParseResult FIRStmtParser::parseSimpleStmt(unsigned stmtIndent) {
  locationProcessor.startStatement();
  auto result = parseSimpleStmtImpl(stmtIndent);
  locationProcessor.endStatement(*this);
  return result;
}

/// simple_stmt ::= stmt
///
/// stmt ::= attach
///      ::= memport
///      ::= printf
///      ::= skip
///      ::= stop
///      ::= when
///      ::= leading-exp-stmt
///      ::= define
///      ::= propassign
///
/// stmt ::= instance
///      ::= cmem | smem | mem
///      ::= node | wire
///      ::= register
///
ParseResult FIRStmtParser::parseSimpleStmtImpl(unsigned stmtIndent) {
  auto kind = getToken().getKind();
  /// Massage the kind based on the FIRRTL Version.
  switch (kind) {
  case FIRToken::kw_invalidate:
  case FIRToken::kw_connect:
  case FIRToken::kw_regreset:
    /// The "invalidate", "connect", and "regreset" keywords were added
    /// in 3.0.0.
    if (FIRVersion::compare(version, FIRVersion({3, 0, 0})) < 0)
      kind = FIRToken::identifier;
    break;
  default:
    break;
  };
  switch (kind) {
  // Statements.
  case FIRToken::kw_attach:
    return parseAttach();
  case FIRToken::kw_infer:
    return parseMemPort(MemDirAttr::Infer);
  case FIRToken::kw_read:
    return parseMemPort(MemDirAttr::Read);
  case FIRToken::kw_write:
    return parseMemPort(MemDirAttr::Write);
  case FIRToken::kw_rdwr:
    return parseMemPort(MemDirAttr::ReadWrite);
  case FIRToken::kw_connect:
    return parseConnect();
  case FIRToken::kw_propassign:
    if (FIRVersion::compare(version, FIRVersion({3, 1, 0})) < 0)
      return emitError() << "unexpected token: Properties are a FIRRTL 3.1.0+ "
                            "feature, but the specified FIRRTL version was "
                         << version;
    return parsePropAssign();
  case FIRToken::kw_invalidate:
    return parseInvalidate();
  case FIRToken::lp_printf:
    return parsePrintf();
  case FIRToken::kw_skip:
    return parseSkip();
  case FIRToken::lp_stop:
    return parseStop();
  case FIRToken::lp_assert:
    return parseAssert();
  case FIRToken::lp_assume:
    return parseAssume();
  case FIRToken::lp_cover:
    return parseCover();
  case FIRToken::kw_when:
    return parseWhen(stmtIndent);
  case FIRToken::kw_match:
    return parseMatch(stmtIndent);
  case FIRToken::kw_define:
    return parseRefDefine();
  case FIRToken::lp_force:
    return parseRefForce();
  case FIRToken::lp_force_initial:
    return parseRefForceInitial();
  case FIRToken::lp_release:
    return parseRefRelease();
  case FIRToken::lp_release_initial:
    return parseRefReleaseInitial();
  case FIRToken::kw_group:
    if (FIRVersion::compare(version, FIRVersion({3, 1, 0})) < 0)
      return emitError()
             << "unexpected token: optional groups are a FIRRTL 3.1.0+ "
                "feature, but the specified FIRRTL version was "
             << version;
    return parseGroup(stmtIndent);

  default: {
    // Statement productions that start with an expression.
    Value lhs;
    if (parseExpLeadingStmt(lhs, "unexpected token in module"))
      return failure();
    // We use parseExp in a special mode that can complete the entire stmt at
    // once in unusual cases.  If this happened, then we are done.
    if (!lhs)
      return success();

    return parseLeadingExpStmt(lhs);
  }

    // Declarations
  case FIRToken::kw_inst:
    return parseInstance();
  case FIRToken::kw_cmem:
    return parseCombMem();
  case FIRToken::kw_smem:
    return parseSeqMem();
  case FIRToken::kw_mem:
    return parseMem(stmtIndent);
  case FIRToken::kw_node:
    return parseNode();
  case FIRToken::kw_wire:
    return parseWire();
  case FIRToken::kw_reg:
    return parseRegister(stmtIndent);
  case FIRToken::kw_regreset:
    return parseRegisterWithReset();
  }
}

ParseResult FIRStmtParser::parseSubBlock(Block &blockToInsertInto,
                                         unsigned indent,
                                         SymbolRefAttr groupSym) {
  // Declarations within the suite are scoped to within the suite.
  auto suiteScope = std::make_unique<FIRModuleContext::ContextScope>(
      moduleContext, &blockToInsertInto);

  // After parsing the when region, we can release any new entries in
  // unbundledValues since the symbol table entries that refer to them will be
  // gone.
  UnbundledValueRestorer x(moduleContext.unbundledValues);

  // We parse the substatements into their own parser, so they get inserted
  // into the specified 'when' region.
  auto subParser = std::make_unique<FIRStmtParser>(
      blockToInsertInto, moduleContext, modNameSpace, version, groupSym);

  // Figure out whether the body is a single statement or a nested one.
  auto stmtIndent = getIndentation();

  // Parsing a single statment is straightforward.
  if (!stmtIndent.has_value())
    return subParser->parseSimpleStmt(indent);

  if (*stmtIndent <= indent)
    return emitError("statement must be indented more than previous statement"),
           failure();

  // Parse a block of statements that are indented more than the when.
  return subParser->parseSimpleStmtBlock(indent);
}

/// attach ::= 'attach' '(' exp+ ')' info?
ParseResult FIRStmtParser::parseAttach() {
  auto startTok = consumeToken(FIRToken::kw_attach);

  // If this was actually the start of a connect or something else handle that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  if (parseToken(FIRToken::l_paren, "expected '(' after attach"))
    return failure();

  SmallVector<Value, 4> operands;
  do {
    operands.push_back({});
    if (parseExp(operands.back(), "expected operand in attach"))
      return failure();
  } while (!consumeIf(FIRToken::r_paren));

  if (parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  builder.create<AttachOp>(operands);
  return success();
}

/// stmt ::= mdir 'mport' id '=' id '[' exp ']' exp info?
/// mdir ::= 'infer' | 'read' | 'write' | 'rdwr'
///
ParseResult FIRStmtParser::parseMemPort(MemDirAttr direction) {
  auto startTok = consumeToken();
  auto startLoc = startTok.getLoc();

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  StringRef memName;
  SymbolValueEntry memorySym;
  Value memory, indexExp, clock;
  if (parseToken(FIRToken::kw_mport, "expected 'mport' in memory port") ||
      parseId(id, "expected result name") ||
      parseToken(FIRToken::equal, "expected '=' in memory port") ||
      parseId(memName, "expected memory name") ||
      moduleContext.lookupSymbolEntry(memorySym, memName, startLoc) ||
      moduleContext.resolveSymbolEntry(memory, memorySym, startLoc) ||
      parseToken(FIRToken::l_square, "expected '[' in memory port") ||
      parseExp(indexExp, "expected index expression") ||
      parseToken(FIRToken::r_square, "expected ']' in memory port") ||
      parseExp(clock, "expected clock expression") || parseOptionalInfo())
    return failure();

  auto memVType = type_dyn_cast<CMemoryType>(memory.getType());
  if (!memVType)
    return emitError(startLoc,
                     "memory port should have behavioral memory type");
  auto resultType = memVType.getElementType();

  ArrayAttr annotations = getConstants().emptyArrayAttr;
  locationProcessor.setLoc(startLoc);

  // Create the memory port at the location of the cmemory.
  Value memoryPort, memoryData;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfterValue(memory);
    auto memoryPortOp = builder.create<MemoryPortOp>(
        resultType, CMemoryPortType::get(getContext()), memory, direction, id,
        annotations);
    memoryData = memoryPortOp.getResult(0);
    memoryPort = memoryPortOp.getResult(1);
  }

  // Create a memory port access in the current scope.
  builder.create<MemoryPortAccessOp>(memoryPort, indexExp, clock);

  return moduleContext.addSymbolEntry(id, memoryData, startLoc, true);
}

/// printf ::= 'printf(' exp exp StringLit exp* ')' name? info?
ParseResult FIRStmtParser::parsePrintf() {
  auto startTok = consumeToken(FIRToken::lp_printf);

  Value clock, condition;
  StringRef formatString;
  if (parseExp(clock, "expected clock expression in printf") ||
      parseExp(condition, "expected condition in printf") ||
      parseGetSpelling(formatString) ||
      parseToken(FIRToken::string, "expected format string in printf"))
    return failure();

  SmallVector<Value, 4> operands;
  while (!consumeIf(FIRToken::r_paren)) {
    operands.push_back({});
    if (parseExp(operands.back(), "expected operand in printf"))
      return failure();
  }

  StringAttr name;
  if (parseOptionalName(name))
    return failure();

  if (parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  auto formatStrUnescaped = FIRToken::getStringValue(formatString);
  builder.create<PrintFOp>(clock, condition,
                           builder.getStringAttr(formatStrUnescaped), operands,
                           name);
  return success();
}

/// skip ::= 'skip' info?
ParseResult FIRStmtParser::parseSkip() {
  auto startTok = consumeToken(FIRToken::kw_skip);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  if (parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  builder.create<SkipOp>();
  return success();
}

/// stop ::= 'stop(' exp exp intLit ')' info?
ParseResult FIRStmtParser::parseStop() {
  auto startTok = consumeToken(FIRToken::lp_stop);

  Value clock, condition;
  int64_t exitCode;
  StringAttr name;
  if (parseExp(clock, "expected clock expression in 'stop'") ||
      parseExp(condition, "expected condition in 'stop'") ||
      parseIntLit(exitCode, "expected exit code in 'stop'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'stop'") ||
      parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  builder.create<StopOp>(clock, condition, builder.getI32IntegerAttr(exitCode),
                         name);
  return success();
}

/// assert ::= 'assert(' exp exp exp StringLit ')' info?
ParseResult FIRStmtParser::parseAssert() {
  auto startTok = consumeToken(FIRToken::lp_assert);

  Value clock, predicate, enable;
  StringRef message;
  StringAttr name;
  if (parseExp(clock, "expected clock expression in 'assert'") ||
      parseExp(predicate, "expected predicate in 'assert'") ||
      parseExp(enable, "expected enable in 'assert'") ||
      parseGetSpelling(message) ||
      parseToken(FIRToken::string, "expected message in 'assert'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'assert'") ||
      parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  auto messageUnescaped = FIRToken::getStringValue(message);
  builder.create<AssertOp>(clock, predicate, enable, messageUnescaped,
                           ValueRange{}, name.getValue());
  return success();
}

/// assume ::= 'assume(' exp exp exp StringLit ')' info?
ParseResult FIRStmtParser::parseAssume() {
  auto startTok = consumeToken(FIRToken::lp_assume);

  Value clock, predicate, enable;
  StringRef message;
  StringAttr name;
  if (parseExp(clock, "expected clock expression in 'assume'") ||
      parseExp(predicate, "expected predicate in 'assume'") ||
      parseExp(enable, "expected enable in 'assume'") ||
      parseGetSpelling(message) ||
      parseToken(FIRToken::string, "expected message in 'assume'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'assume'") ||
      parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  auto messageUnescaped = FIRToken::getStringValue(message);
  builder.create<AssumeOp>(clock, predicate, enable, messageUnescaped,
                           ValueRange{}, name.getValue());
  return success();
}

/// cover ::= 'cover(' exp exp exp StringLit ')' info?
ParseResult FIRStmtParser::parseCover() {
  auto startTok = consumeToken(FIRToken::lp_cover);

  Value clock, predicate, enable;
  StringRef message;
  StringAttr name;
  if (parseExp(clock, "expected clock expression in 'cover'") ||
      parseExp(predicate, "expected predicate in 'cover'") ||
      parseExp(enable, "expected enable in 'cover'") ||
      parseGetSpelling(message) ||
      parseToken(FIRToken::string, "expected message in 'cover'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'cover'") ||
      parseOptionalName(name) || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  auto messageUnescaped = FIRToken::getStringValue(message);
  builder.create<CoverOp>(clock, predicate, enable, messageUnescaped,
                          ValueRange{}, name.getValue());
  return success();
}

/// when  ::= 'when' exp ':' info? suite? ('else' ( when | ':' info? suite?)
/// )? suite ::= simple_stmt | INDENT simple_stmt+ DEDENT
ParseResult FIRStmtParser::parseWhen(unsigned whenIndent) {
  auto startTok = consumeToken(FIRToken::kw_when);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  Value condition;
  if (parseExp(condition, "expected condition in 'when'") ||
      parseToken(FIRToken::colon, "expected ':' in when") ||
      parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  // Create the IR representation for the when.
  auto whenStmt = builder.create<WhenOp>(condition, /*createElse*/ false);

  // Parse the 'then' body into the 'then' region.
  if (parseSubBlock(whenStmt.getThenBlock(), whenIndent, groupSym))
    return failure();

  // If the else is present, handle it otherwise we're done.
  if (getToken().isNot(FIRToken::kw_else))
    return success();

  // If the 'else' is less indented than the when, then it must belong to some
  // containing 'when'.
  auto elseIndent = getIndentation();
  if (elseIndent && *elseIndent < whenIndent)
    return success();

  consumeToken(FIRToken::kw_else);

  // Create an else block to parse into.
  whenStmt.createElseRegion();

  // If we have the ':' form, then handle it.

  // Syntactic shorthand 'else when'. This uses the same indentation level as
  // the outer 'when'.
  if (getToken().is(FIRToken::kw_when)) {
    // We create a sub parser for the else block.
    auto subParser =
        std::make_unique<FIRStmtParser>(whenStmt.getElseBlock(), moduleContext,
                                        modNameSpace, version, groupSym);

    return subParser->parseSimpleStmt(whenIndent);
  }

  // Parse the 'else' body into the 'else' region.
  LocationAttr elseLoc; // ignore the else locator.
  if (parseToken(FIRToken::colon, "expected ':' after 'else'") ||
      parseOptionalInfoLocator(elseLoc) ||
      parseSubBlock(whenStmt.getElseBlock(), whenIndent, groupSym))
    return failure();

  // TODO(firrtl spec): There is no reason for the 'else :' grammar to take an
  // info.  It doesn't appear to be generated either.
  return success();
}

/// enum-exp ::= enum-type '(' Id ( ',' exp )? ')'
ParseResult FIRStmtParser::parseEnumExp(Value &value) {
  auto startLoc = getToken().getLoc();
  FIRRTLType type;
  if (parseEnumType(type))
    return failure();

  // Check that the input type is a legal enumeration.
  auto enumType = type_dyn_cast<FEnumType>(type);
  if (!enumType)
    return emitError(startLoc,
                     "expected enumeration type in enumeration expression");

  StringRef tag;
  if (parseToken(FIRToken::l_paren, "expected '(' in enumeration expression") ||
      parseId(tag, "expected enumeration tag"))
    return failure();

  Value input;
  if (consumeIf(FIRToken::r_paren)) {
    // If the payload is not specified, we create a 0 bit unsigned integer
    // constant.
    auto type = IntType::get(builder.getContext(), false, 0);
    Type attrType = IntegerType::get(getContext(), 0, IntegerType::Unsigned);
    auto attr = builder.getIntegerAttr(attrType, APInt(0, 0, false));
    input = builder.create<ConstantOp>(type, attr);
  } else {
    // Otherwise we parse an expression.
    if (parseExp(input, "expected expression in enumeration value") ||
        parseToken(FIRToken::r_paren, "expected closing ')'"))
      return failure();
  }

  locationProcessor.setLoc(startLoc);
  value = builder.create<FEnumCreateOp>(enumType, tag, input);
  return success();
}

/// match ::= 'match' exp ':' info?
///             (INDENT ( Id ( '(' Id ')' )? ':'
///               (INDENT simple_stmt* DEDENT )?
///             )* DEDENT)?
ParseResult FIRStmtParser::parseMatch(unsigned matchIndent) {
  auto startTok = consumeToken(FIRToken::kw_match);

  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  Value input;
  if (parseExp(input, "expected expression in 'match'") ||
      parseToken(FIRToken::colon, "expected ':' in 'match'") ||
      parseOptionalInfo())
    return failure();

  auto enumType = type_dyn_cast<FEnumType>(input.getType());
  if (!enumType)
    return mlir::emitError(
               input.getLoc(),
               "expected enumeration type for 'match' statement, but got ")
           << input.getType();

  locationProcessor.setLoc(startTok.getLoc());

  SmallVector<Attribute> tags;
  SmallVector<std::unique_ptr<Region>> regions;
  while (true) {
    auto tagLoc = getToken().getLoc();

    // Only consume the keyword if the indentation is correct.
    auto caseIndent = getIndentation();
    if (!caseIndent || *caseIndent <= matchIndent)
      break;

    // Parse the tag.
    StringRef tagSpelling;
    if (parseId(tagSpelling, "expected enumeration tag in match statement"))
      return failure();
    auto tagIndex = enumType.getElementIndex(tagSpelling);
    if (!tagIndex)
      return emitError(tagLoc, "tag ")
             << tagSpelling << " not a member of enumeration " << enumType;
    auto tag = IntegerAttr::get(IntegerType::get(getContext(), 32), *tagIndex);
    tags.push_back(tag);

    // Add a new case to the match operation.
    auto *caseBlock = &regions.emplace_back(new Region)->emplaceBlock();

    // Declarations are scoped to the case.
    FIRModuleContext::ContextScope scope(moduleContext, caseBlock);

    // After parsing the region, we can release any new entries in
    // unbundledValues since the symbol table entries that refer to them will be
    // gone.
    UnbundledValueRestorer x(moduleContext.unbundledValues);

    // Parse the argument.
    if (consumeIf(FIRToken::l_paren)) {
      StringAttr identifier;
      if (parseId(identifier, "expected identifier for 'case' binding"))
        return failure();

      // Add an argument to the block.
      auto dataType = enumType.getElementType(*tagIndex);
      caseBlock->addArgument(dataType, LocWithInfo(tagLoc, this).getLoc());

      if (moduleContext.addSymbolEntry(identifier, caseBlock->getArgument(0),
                                       startTok.getLoc()))
        return failure();

      if (parseToken(FIRToken::r_paren, "expected ')' in match statement case"))
        return failure();

    } else {
      auto dataType = IntType::get(builder.getContext(), false, 0);
      caseBlock->addArgument(dataType, LocWithInfo(tagLoc, this).getLoc());
    }

    if (parseToken(FIRToken::colon, "expected ':' in match statement case"))
      return failure();

    // Parse a block of statements that are indented more than the case.
    auto subParser = std::make_unique<FIRStmtParser>(
        *caseBlock, moduleContext, modNameSpace, version, groupSym);
    if (subParser->parseSimpleStmtBlock(*caseIndent))
      return failure();
  }

  builder.create<MatchOp>(input, ArrayAttr::get(getContext(), tags), regions);
  return success();
}

/// ref_expr ::= probe | rwprobe | static_reference
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRStmtParser::parseRefExp(Value &result, const Twine &message) {
  auto token = getToken().getKind();
  if (token == FIRToken::lp_probe)
    return parseProbe(result);
  if (token == FIRToken::lp_rwprobe)
    return parseRWProbe(result);

  // Default to parsing as static reference expression.
  // Don't check token kind, we need to support literal_identifier and keywords,
  // let parseId handle this.
  return parseStaticRefExp(result, message);
}

/// static_reference ::= id
///                  ::= static_reference '.' id
///                  ::= static_reference '[' int ']'
// NOLINTNEXTLINE(misc-no-recursion)
ParseResult FIRStmtParser::parseStaticRefExp(Value &result,
                                             const Twine &message) {
  auto parseIdOrInstance = [&]() -> ParseResult {
    StringRef id;
    auto loc = getToken().getLoc();
    SymbolValueEntry symtabEntry;
    if (parseId(id, message) ||
        moduleContext.lookupSymbolEntry(symtabEntry, id, loc))
      return failure();

    // If we looked up a normal value, then we're done.
    if (!moduleContext.resolveSymbolEntry(result, symtabEntry, loc, false))
      return success();

    assert(symtabEntry.is<UnbundledID>() && "should be an instance");

    // Handle the normal "instance.x" reference.
    StringRef fieldName;
    return failure(
        parseToken(FIRToken::period, "expected '.' in field reference") ||
        parseFieldId(fieldName, "expected field name") ||
        moduleContext.resolveSymbolEntry(result, symtabEntry, fieldName, loc));
  };
  return failure(parseIdOrInstance() ||
                 parseOptionalExpPostscript(result, false));
}

/// define ::= 'define' static_reference '=' ref_expr info?
ParseResult FIRStmtParser::parseRefDefine() {
  auto startTok = consumeToken(FIRToken::kw_define);

  Value src, target;
  if (parseStaticRefExp(target,
                        "expected static reference expression in 'define'") ||
      parseToken(FIRToken::equal,
                 "expected '=' after define reference expression") ||
      parseRefExp(src, "expected reference expression in 'define'") ||
      parseOptionalInfo())
    return failure();

  // Check reference expressions are of reference type.
  if (!type_isa<RefType>(target.getType()))
    return emitError(startTok.getLoc(), "expected reference-type expression in "
                                        "'define' target (LHS), got ")
           << target.getType();
  if (!type_isa<RefType>(src.getType()))
    return emitError(startTok.getLoc(), "expected reference-type expression in "
                                        "'define' source (RHS), got ")
           << src.getType();

  // static_reference doesn't differentiate which can be ref.sub'd, so check
  // this explicitly:
  if (isa_and_nonnull<RefSubOp>(target.getDefiningOp()))
    return emitError(startTok.getLoc(),
                     "cannot define into a sub-element of a reference");

  locationProcessor.setLoc(startTok.getLoc());

  if (!areTypesRefCastable(target.getType(), src.getType()))
    return emitError(startTok.getLoc(), "cannot define reference of type ")
           << target.getType() << " with incompatible reference of type "
           << src.getType();

  emitConnect(builder, target, src);

  return success();
}

/// read ::= '(' ref_expr ')'
/// XXX: spec says static_reference, allow ref_expr anyway for read(probe(x)).
ParseResult FIRStmtParser::parseRefRead(Value &result) {
  auto startTok = consumeToken(FIRToken::lp_read);

  Value ref;
  if (parseRefExp(ref, "expected reference expression in 'read'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'read'"))
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Check argument is a ref-type value.
  if (!type_isa<RefType>(ref.getType()))
    return emitError(startTok.getLoc(),
                     "expected reference-type expression in 'read', got ")
           << ref.getType();

  result = builder.create<RefResolveOp>(ref);

  return success();
}

/// probe ::= 'probe' '(' static_ref ')'
ParseResult FIRStmtParser::parseProbe(Value &result) {
  auto startTok = consumeToken(FIRToken::lp_probe);

  Value staticRef;
  if (parseStaticRefExp(staticRef,
                        "expected static reference expression in 'probe'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'probe'"))
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Check probe expression is base-type.
  if (!type_isa<FIRRTLBaseType>(staticRef.getType()))
    return emitError(startTok.getLoc(),
                     "expected base-type expression in 'probe', got ")
           << staticRef.getType();

  // Check for other unsupported reference sources.
  // TODO: Add to ref.send verifier / inferReturnTypes.
  if (isa_and_nonnull<MemOp, CombMemOp, SeqMemOp, MemoryPortOp,
                      MemoryDebugPortOp, MemoryPortAccessOp>(
          staticRef.getDefiningOp()))
    return emitError(startTok.getLoc(), "cannot probe memories or their ports");

  result = builder.create<RefSendOp>(staticRef);

  return success();
}

/// rwprobe ::= 'rwprobe' '(' static_ref ')'
ParseResult FIRStmtParser::parseRWProbe(Value &result) {
  auto startTok = consumeToken(FIRToken::lp_rwprobe);

  Value staticRef;
  if (parseStaticRefExp(staticRef,
                        "expected static reference expression in 'rwprobe'") ||
      parseToken(FIRToken::r_paren, "expected ')' in 'rwprobe'"))
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Checks:
  // Not public port (verifier)

  // Check probe expression is base-type.
  auto targetType = type_dyn_cast<FIRRTLBaseType>(staticRef.getType());
  if (!targetType)
    return emitError(startTok.getLoc(),
                     "expected base-type expression in 'rwprobe', got ")
           << staticRef.getType();

  auto fieldRef = getFieldRefFromValue(staticRef);
  auto target = fieldRef.getValue();

  // Ports are handled differently, emit a RWProbeOp with inner symbol.
  if (auto arg = dyn_cast<BlockArgument>(target)) {
    // Check target type.  Replicate inference/verification logic.
    if (targetType.hasUninferredWidth() || targetType.hasUninferredReset())
      return emitError(startTok.getLoc(),
                       "must have known width or concrete reset type in type ")
             << targetType;
    auto forceableType =
        firrtl::detail::getForceableResultType(true, targetType);
    if (!forceableType)
      return emitError(startTok.getLoc(), "cannot force target of type ")
             << targetType;

    // Get InnerRef for target field.
    auto mod = cast<FModuleOp>(arg.getOwner()->getParentOp());
    auto sym = getInnerRefTo(
        hw::InnerSymTarget(arg.getArgNumber(), mod, fieldRef.getFieldID()),
        [&](FModuleOp mod) -> ModuleNamespace & { return modNameSpace; });
    result = builder.create<RWProbeOp>(sym, targetType);
    return success();
  }

  auto *definingOp = target.getDefiningOp();
  if (!definingOp)
    return emitError(startTok.getLoc(),
                     "rwprobe value must be defined by an operation");

  if (isa<MemOp, CombMemOp, SeqMemOp, MemoryPortOp, MemoryDebugPortOp,
          MemoryPortAccessOp>(definingOp))
    return emitError(startTok.getLoc(), "cannot probe memories or their ports");

  auto forceable = dyn_cast<Forceable>(definingOp);
  if (!forceable || !forceable.isForceable() /* e.g., is/has const type*/)
    return emitError(startTok.getLoc(), "rwprobe target not forceable")
        .attachNote(definingOp->getLoc());

  // TODO: do the ref.sub work while parsing the static expression.
  result =
      getValueByFieldID(builder, forceable.getDataRef(), fieldRef.getFieldID());

  return success();
}

/// force ::= 'force(' exp exp ref_expr exp ')' info?
ParseResult FIRStmtParser::parseRefForce() {
  auto startTok = consumeToken(FIRToken::lp_force);

  Value clock, pred, dest, src;
  if (parseExp(clock, "expected clock expression in force") ||
      parseExp(pred, "expected predicate expression in force") ||
      parseRefExp(dest, "expected destination reference expression in force") ||
      parseExp(src, "expected source expression in force") ||
      parseToken(FIRToken::r_paren, "expected ')' in force") ||
      parseOptionalInfo())
    return failure();

  // Check reference expression is of reference type.
  auto ref = type_dyn_cast<RefType>(dest.getType());
  if (!ref || !ref.getForceable())
    return emitError(
               startTok.getLoc(),
               "expected rwprobe-type expression for force destination, got ")
           << dest.getType();
  auto srcBaseType = type_dyn_cast<FIRRTLBaseType>(src.getType());
  if (!srcBaseType)
    return emitError(startTok.getLoc(),
                     "expected base-type for force source, got ")
           << src.getType();

  locationProcessor.setLoc(startTok.getLoc());

  // Cast ref to accomodate uninferred sources.
  auto noConstSrcType = srcBaseType.getAllConstDroppedType();
  if (noConstSrcType != ref.getType()) {
    // Try to cast destination to rwprobe of source type (dropping const).
    auto compatibleRWProbe = RefType::get(noConstSrcType, true);
    if (areTypesRefCastable(compatibleRWProbe, ref))
      dest = builder.create<RefCastOp>(compatibleRWProbe, dest);
  }

  builder.create<RefForceOp>(clock, pred, dest, src);

  return success();
}

/// force_initial ::= 'force_initial(' ref_expr exp ')' info?
ParseResult FIRStmtParser::parseRefForceInitial() {
  auto startTok = consumeToken(FIRToken::lp_force_initial);

  Value dest, src;
  if (parseRefExp(
          dest, "expected destination reference expression in force_initial") ||
      parseExp(src, "expected source expression in force_initial") ||
      parseToken(FIRToken::r_paren, "expected ')' in force_initial") ||
      parseOptionalInfo())
    return failure();

  // Check reference expression is of reference type.
  auto ref = type_dyn_cast<RefType>(dest.getType());
  if (!ref || !ref.getForceable())
    return emitError(startTok.getLoc(), "expected rwprobe-type expression for "
                                        "force_initial destination, got ")
           << dest.getType();
  auto srcBaseType = type_dyn_cast<FIRRTLBaseType>(src.getType());
  if (!srcBaseType)
    return emitError(startTok.getLoc(),
                     "expected base-type expression for force_initial "
                     "source, got ")
           << src.getType();

  locationProcessor.setLoc(startTok.getLoc());

  // Cast ref to accomodate uninferred sources.
  auto noConstSrcType = srcBaseType.getAllConstDroppedType();
  if (noConstSrcType != ref.getType()) {
    // Try to cast destination to rwprobe of source type (dropping const).
    auto compatibleRWProbe = RefType::get(noConstSrcType, true);
    if (areTypesRefCastable(compatibleRWProbe, ref))
      dest = builder.create<RefCastOp>(compatibleRWProbe, dest);
  }

  auto value = APInt::getAllOnes(1);
  auto type = UIntType::get(builder.getContext(), 1);
  auto attr = builder.getIntegerAttr(IntegerType::get(type.getContext(),
                                                      value.getBitWidth(),
                                                      IntegerType::Unsigned),
                                     value);
  auto pred = moduleContext.getCachedConstantInt(builder, attr, type, value);
  builder.create<RefForceInitialOp>(pred, dest, src);

  return success();
}

/// release ::= 'release(' exp exp ref_expr ')' info?
ParseResult FIRStmtParser::parseRefRelease() {
  auto startTok = consumeToken(FIRToken::lp_release);

  Value clock, pred, dest;
  if (parseExp(clock, "expected clock expression in release") ||
      parseExp(pred, "expected predicate expression in release") ||
      parseRefExp(dest,
                  "expected destination reference expression in release") ||
      parseToken(FIRToken::r_paren, "expected ')' in release") ||
      parseOptionalInfo())
    return failure();

  // Check reference expression is of reference type.
  if (auto ref = type_dyn_cast<RefType>(dest.getType());
      !ref || !ref.getForceable())
    return emitError(
               startTok.getLoc(),
               "expected rwprobe-type expression for release destination, got ")
           << dest.getType();

  locationProcessor.setLoc(startTok.getLoc());

  builder.create<RefReleaseOp>(clock, pred, dest);

  return success();
}

/// release_initial ::= 'release_initial(' ref_expr ')' info?
ParseResult FIRStmtParser::parseRefReleaseInitial() {
  auto startTok = consumeToken(FIRToken::lp_release_initial);

  Value dest;
  if (parseRefExp(
          dest,
          "expected destination reference expression in release_initial") ||
      parseToken(FIRToken::r_paren, "expected ')' in release_initial") ||
      parseOptionalInfo())
    return failure();

  // Check reference expression is of reference type.
  if (auto ref = type_dyn_cast<RefType>(dest.getType());
      !ref || !ref.getForceable())
    return emitError(startTok.getLoc(), "expected rwprobe-type expression for "
                                        "release_initial destination, got ")
           << dest.getType();

  locationProcessor.setLoc(startTok.getLoc());

  auto value = APInt::getAllOnes(1);
  auto type = UIntType::get(builder.getContext(), 1);
  auto attr = builder.getIntegerAttr(IntegerType::get(type.getContext(),
                                                      value.getBitWidth(),
                                                      IntegerType::Unsigned),
                                     value);
  auto pred = moduleContext.getCachedConstantInt(builder, attr, type, value);
  builder.create<RefReleaseInitialOp>(pred, dest);

  return success();
}

/// connect ::= 'connect' expr expr
ParseResult FIRStmtParser::parseConnect() {
  auto startTok = consumeToken(FIRToken::kw_connect);
  auto loc = startTok.getLoc();

  Value lhs, rhs;
  if (parseExp(lhs, "expected connect expression") ||
      parseExp(rhs, "expected connect expression") || parseOptionalInfo())
    return failure();

  auto lhsType = type_dyn_cast<FIRRTLBaseType>(lhs.getType());
  auto rhsType = type_dyn_cast<FIRRTLBaseType>(rhs.getType());
  if (!lhsType || !rhsType)
    return emitError(loc, "cannot connect reference or property types");
  // TODO: Once support lands for agg-of-ref, add test for this check!
  if (lhsType.containsReference() || rhsType.containsReference())
    return emitError(loc, "cannot connect types containing references");

  if (!areTypesEquivalent(lhsType, rhsType))
    return emitError(loc, "cannot connect non-equivalent type ")
           << rhsType << " to " << lhsType;

  locationProcessor.setLoc(loc);
  emitConnect(builder, lhs, rhs);
  return success();
}

/// propassign ::= 'propassign' expr expr
ParseResult FIRStmtParser::parsePropAssign() {
  auto startTok = consumeToken(FIRToken::kw_propassign);
  auto loc = startTok.getLoc();

  Value lhs, rhs;
  if (parseExp(lhs, "expected propassign expression") ||
      parseExp(rhs, "expected propassign expression") || parseOptionalInfo())
    return failure();

  auto lhsType = type_dyn_cast<PropertyType>(lhs.getType());
  auto rhsType = type_dyn_cast<PropertyType>(rhs.getType());
  if (!lhsType || !rhsType)
    return emitError(loc, "can only propassign property types");
  if (lhsType != rhsType)
    return emitError(loc, "cannot propassign non-equivalent type ")
           << rhsType << " to " << lhsType;
  locationProcessor.setLoc(loc);
  builder.create<PropAssignOp>(lhs, rhs);
  return success();
}

/// invalidate ::= 'invalidate' expr
ParseResult FIRStmtParser::parseInvalidate() {
  auto startTok = consumeToken(FIRToken::kw_invalidate);

  Value lhs;
  if (parseExp(lhs, "expected connect expression") || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());
  emitInvalidate(lhs);
  return success();
}

ParseResult FIRStmtParser::parseGroup(unsigned indent) {

  auto startTok = consumeToken(FIRToken::kw_group);
  auto loc = startTok.getLoc();

  StringRef id;
  if (parseId(id, "expected group identifer") ||
      parseToken(FIRToken::colon, "expected ':' at end of group") ||
      parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(loc);

  StringRef rootGroup;
  SmallVector<FlatSymbolRefAttr> nestedGroups;
  if (!groupSym) {
    rootGroup = id;
  } else {
    rootGroup = groupSym.getRootReference();
    auto nestedRefs = groupSym.getNestedReferences();
    nestedGroups.append(nestedRefs.begin(), nestedRefs.end());
    nestedGroups.push_back(FlatSymbolRefAttr::get(builder.getContext(), id));
  }

  auto groupOp = builder.create<GroupOp>(
      SymbolRefAttr::get(builder.getContext(), rootGroup, nestedGroups));
  groupOp->getRegion(0).push_back(new Block());

  if (getIndentation() > indent)
    if (parseSubBlock(groupOp.getRegion().front(), indent,
                      groupOp.getGroupName()))
      return failure();

  return success();
}

/// leading-exp-stmt ::= exp '<=' exp info?
///                  ::= exp '<-' exp info?
///                  ::= exp 'is' 'invalid' info?
ParseResult FIRStmtParser::parseLeadingExpStmt(Value lhs) {
  auto loc = getToken().getLoc();

  // If 'is' grammar is special.
  if (consumeIf(FIRToken::kw_is)) {
    if (parseToken(FIRToken::kw_invalid, "expected 'invalid'") ||
        parseOptionalInfo())
      return failure();

    locationProcessor.setLoc(loc);
    emitInvalidate(lhs);
    return success();
  }

  auto kind = getToken().getKind();
  switch (kind) {
  case FIRToken::less_equal:
    break;
  case FIRToken::less_minus:
    // Partial connect ("<-") was removed in FIRRTL version 2.0.0.
    if (FIRVersion::compare(version, FIRVersion({2, 0, 0})) < 0)
      break;
    [[fallthrough]];
  default:
    return emitError() << "unexpected token '" << getToken().getSpelling()
                       << "' in statement",
           failure();
  }
  consumeToken();

  Value rhs;
  if (parseExp(rhs, "unexpected token in statement") || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(loc);

  auto lhsType = type_dyn_cast<FIRRTLBaseType>(lhs.getType());
  auto rhsType = type_dyn_cast<FIRRTLBaseType>(rhs.getType());
  if (!lhsType || !rhsType)
    return emitError(loc, "cannot connect reference or property types");
  // TODO: Once support lands for agg-of-ref, add test for this check!
  if (lhsType.containsReference() || rhsType.containsReference())
    return emitError(loc, "cannot connect types containing references");

  if (kind == FIRToken::less_equal) {
    if (!areTypesEquivalent(lhsType, rhsType))
      return emitError(loc, "cannot connect non-equivalent type ")
             << rhsType << " to " << lhsType;
    emitConnect(builder, lhs, rhs);
  } else {
    assert(kind == FIRToken::less_minus && "unexpected kind");
    if (!areTypesWeaklyEquivalent(lhsType, rhsType))
      return emitError(loc,
                       "cannot partially connect non-weakly-equivalent type ")
             << rhsType << " to " << lhsType;
    emitPartialConnect(builder, lhs, rhs);
  }
  return success();
}

//===-------------------------------
// FIRStmtParser Declaration Parsing

/// instance ::= 'inst' id 'of' id info?
ParseResult FIRStmtParser::parseInstance() {
  auto startTok = consumeToken(FIRToken::kw_inst);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  StringRef moduleName;
  if (parseId(id, "expected instance name") ||
      parseToken(FIRToken::kw_of, "expected 'of' in instance") ||
      parseId(moduleName, "expected module name") || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Look up the module that is being referenced.
  auto circuit =
      builder.getBlock()->getParentOp()->getParentOfType<CircuitOp>();
  auto referencedModule =
      dyn_cast_or_null<FModuleLike>(circuit.lookupSymbol(moduleName));
  if (!referencedModule) {
    emitError(startTok.getLoc(),
              "use of undefined module name '" + moduleName + "' in instance");
    return failure();
  }

  SmallVector<PortInfo> modulePorts = referencedModule.getPorts();

  // Make a bundle of the inputs and outputs of the specified module.
  SmallVector<Type, 4> resultTypes;
  resultTypes.reserve(modulePorts.size());
  SmallVector<std::pair<StringAttr, Type>, 4> resultNamesAndTypes;

  for (auto port : modulePorts) {
    resultTypes.push_back(port.type);
    resultNamesAndTypes.push_back({port.name, port.type});
  }

  auto annotations = getConstants().emptyArrayAttr;
  SmallVector<Attribute, 4> portAnnotations(modulePorts.size(), annotations);

  StringAttr sym = {};
  auto result = builder.create<InstanceOp>(
      referencedModule, id, NameKindEnum::InterestingName,
      annotations.getValue(), portAnnotations, false, sym);

  // Since we are implicitly unbundling the instance results, we need to keep
  // track of the mapping from bundle fields to results in the unbundledValues
  // data structure.  Build our entry now.
  UnbundledValueEntry unbundledValueEntry;
  unbundledValueEntry.reserve(modulePorts.size());
  for (size_t i = 0, e = modulePorts.size(); i != e; ++i)
    unbundledValueEntry.push_back({modulePorts[i].name, result.getResult(i)});

  // Add it to unbundledValues and add an entry to the symbol table to remember
  // it.
  moduleContext.unbundledValues.push_back(std::move(unbundledValueEntry));
  auto entryId = UnbundledID(moduleContext.unbundledValues.size());
  return moduleContext.addSymbolEntry(id, entryId, startTok.getLoc());
}

/// cmem ::= 'cmem' id ':' type info?
ParseResult FIRStmtParser::parseCombMem() {
  // TODO(firrtl spec) cmem is completely undocumented.
  auto startTok = consumeToken(FIRToken::kw_cmem);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  FIRRTLType type;
  if (parseId(id, "expected cmem name") ||
      parseToken(FIRToken::colon, "expected ':' in cmem") ||
      parseType(type, "expected cmem type") || parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Transform the parsed vector type into a memory type.
  auto vectorType = type_dyn_cast<FVectorType>(type);
  if (!vectorType)
    return emitError("cmem requires vector type");

  auto annotations = getConstants().emptyArrayAttr;
  StringAttr sym = {};
  auto result = builder.create<CombMemOp>(
      vectorType.getElementType(), vectorType.getNumElements(), id,
      NameKindEnum::InterestingName, annotations, sym);
  return moduleContext.addSymbolEntry(id, result, startTok.getLoc());
}

/// smem ::= 'smem' id ':' type ruw? info?
ParseResult FIRStmtParser::parseSeqMem() {
  // TODO(firrtl spec) smem is completely undocumented.
  auto startTok = consumeToken(FIRToken::kw_smem);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  FIRRTLType type;
  RUWAttr ruw = RUWAttr::Undefined;

  if (parseId(id, "expected smem name") ||
      parseToken(FIRToken::colon, "expected ':' in smem") ||
      parseType(type, "expected smem type") || parseOptionalRUW(ruw) ||
      parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Transform the parsed vector type into a memory type.
  auto vectorType = type_dyn_cast<FVectorType>(type);
  if (!vectorType)
    return emitError("smem requires vector type");

  auto annotations = getConstants().emptyArrayAttr;
  StringAttr sym = {};
  auto result = builder.create<SeqMemOp>(
      vectorType.getElementType(), vectorType.getNumElements(), ruw, id,
      NameKindEnum::InterestingName, annotations, sym);
  return moduleContext.addSymbolEntry(id, result, startTok.getLoc());
}

/// mem ::= 'mem' id ':' info? INDENT memField* DEDENT
/// memField ::= 'data-type' '=>' type NEWLINE
///          ::= 'depth' '=>' intLit NEWLINE
///          ::= 'read-latency' '=>' intLit NEWLINE
///          ::= 'write-latency' '=>' intLit NEWLINE
///          ::= 'read-under-write' '=>' ruw NEWLINE
///          ::= 'reader' '=>' id+ NEWLINE
///          ::= 'writer' '=>' id+ NEWLINE
///          ::= 'readwriter' '=>' id+ NEWLINE
ParseResult FIRStmtParser::parseMem(unsigned memIndent) {
  auto startTok = consumeToken(FIRToken::kw_mem);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  if (parseId(id, "expected mem name") ||
      parseToken(FIRToken::colon, "expected ':' in mem") || parseOptionalInfo())
    return failure();

  FIRRTLType type;
  int64_t depth = -1, readLatency = -1, writeLatency = -1;
  RUWAttr ruw = RUWAttr::Undefined;

  SmallVector<std::pair<StringAttr, Type>, 4> ports;

  // Parse all the memfield records, which are indented more than the mem.
  while (1) {
    auto nextIndent = getIndentation();
    if (!nextIndent || *nextIndent <= memIndent)
      break;

    auto spelling = getTokenSpelling();
    if (parseToken(FIRToken::identifier, "unexpected token in 'mem'") ||
        parseToken(FIRToken::equal_greater, "expected '=>' in 'mem'"))
      return failure();

    if (spelling == "data-type") {
      if (type)
        return emitError("'mem' type specified multiple times"), failure();

      if (parseType(type, "expected type in data-type declaration"))
        return failure();
      continue;
    }
    if (spelling == "depth") {
      if (parseIntLit(depth, "expected integer in depth specification"))
        return failure();
      continue;
    }
    if (spelling == "read-latency") {
      if (parseIntLit(readLatency, "expected integer latency"))
        return failure();
      continue;
    }
    if (spelling == "write-latency") {
      if (parseIntLit(writeLatency, "expected integer latency"))
        return failure();
      continue;
    }
    if (spelling == "read-under-write") {
      if (getToken().isNot(FIRToken::kw_old, FIRToken::kw_new,
                           FIRToken::kw_undefined))
        return emitError("expected specifier"), failure();

      if (parseOptionalRUW(ruw))
        return failure();
      continue;
    }

    MemOp::PortKind portKind;
    if (spelling == "reader")
      portKind = MemOp::PortKind::Read;
    else if (spelling == "writer")
      portKind = MemOp::PortKind::Write;
    else if (spelling == "readwriter")
      portKind = MemOp::PortKind::ReadWrite;
    else
      return emitError("unexpected field in 'mem' declaration"), failure();

    StringRef portName;
    if (parseId(portName, "expected port name"))
      return failure();
    auto baseType = type_dyn_cast<FIRRTLBaseType>(type);
    if (!baseType)
      return emitError("unexpected type, must be base type");
    ports.push_back({builder.getStringAttr(portName),
                     MemOp::getTypeForPort(depth, baseType, portKind)});

    while (!getIndentation().has_value()) {
      if (parseId(portName, "expected port name"))
        return failure();
      ports.push_back({builder.getStringAttr(portName),
                       MemOp::getTypeForPort(depth, baseType, portKind)});
    }
  }

  // The FIRRTL dialect requires mems to have at least one port.  Since portless
  // mems can never be referenced, it is always safe to drop them.
  if (ports.empty())
    return success();

  // Canonicalize the ports into alphabetical order.
  // TODO: Move this into MemOp construction/canonicalization.
  llvm::array_pod_sort(ports.begin(), ports.end(),
                       [](const std::pair<StringAttr, Type> *lhs,
                          const std::pair<StringAttr, Type> *rhs) -> int {
                         return lhs->first.getValue().compare(
                             rhs->first.getValue());
                       });

  auto annotations = getConstants().emptyArrayAttr;
  SmallVector<Attribute, 4> resultNames;
  SmallVector<Type, 4> resultTypes;
  SmallVector<Attribute, 4> resultAnnotations;
  for (auto p : ports) {
    resultNames.push_back(p.first);
    resultTypes.push_back(p.second);
    resultAnnotations.push_back(annotations);
  }

  locationProcessor.setLoc(startTok.getLoc());

  auto result = builder.create<MemOp>(
      resultTypes, readLatency, writeLatency, depth, ruw,
      builder.getArrayAttr(resultNames), id, NameKindEnum::InterestingName,
      annotations, builder.getArrayAttr(resultAnnotations), hw::InnerSymAttr(),
      MemoryInitAttr(), StringAttr());

  UnbundledValueEntry unbundledValueEntry;
  unbundledValueEntry.reserve(result.getNumResults());
  for (size_t i = 0, e = result.getNumResults(); i != e; ++i)
    unbundledValueEntry.push_back({resultNames[i], result.getResult(i)});

  moduleContext.unbundledValues.push_back(std::move(unbundledValueEntry));
  auto entryID = UnbundledID(moduleContext.unbundledValues.size());
  return moduleContext.addSymbolEntry(id, entryID, startTok.getLoc());
}

/// node ::= 'node' id '=' exp info?
ParseResult FIRStmtParser::parseNode() {
  auto startTok = consumeToken(FIRToken::kw_node);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  Value initializer;
  if (parseId(id, "expected node name") ||
      parseToken(FIRToken::equal, "expected '=' in node") ||
      parseExp(initializer, "expected expression for node") ||
      parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  // Error out in the following conditions:
  //
  //   1. Node type is Analog (at the top level)
  //   2. Node type is not passive under an optional outer flip
  //      (analog field is okay)
  //
  // Note: (1) is more restictive than normal NodeOp verification, but
  // this is added to align with the SFC. (2) is less restrictive than
  // the SFC to accomodate for situations where the node is something
  // weird like a module output or an instance input.
  auto initializerType = type_cast<FIRRTLType>(initializer.getType());
  auto initializerBaseType =
      type_dyn_cast<FIRRTLBaseType>(initializer.getType());
  if (type_isa<AnalogType>(initializerType) ||
      !(initializerBaseType && initializerBaseType.isPassive())) {
    emitError(startTok.getLoc())
        << "Node cannot be analog and must be passive or passive under a flip "
        << initializer.getType();
    return failure();
  }

  auto annotations = getConstants().emptyArrayAttr;
  StringAttr sym = {};

  bool forceable =
      !!firrtl::detail::getForceableResultType(true, initializer.getType());
  auto result =
      builder.create<NodeOp>(initializer, id, NameKindEnum::InterestingName,
                             annotations, sym, forceable);
  return moduleContext.addSymbolEntry(id, result.getResult(),
                                      startTok.getLoc());
}

/// wire ::= 'wire' id ':' type info?
ParseResult FIRStmtParser::parseWire() {
  auto startTok = consumeToken(FIRToken::kw_wire);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  FIRRTLType type;
  if (parseId(id, "expected wire name") ||
      parseToken(FIRToken::colon, "expected ':' in wire") ||
      parseType(type, "expected wire type") || parseOptionalInfo())
    return failure();

  if (!type_isa<FIRRTLBaseType>(type))
    return emitError(startTok.getLoc(), "wire must have base type");

  locationProcessor.setLoc(startTok.getLoc());

  auto annotations = getConstants().emptyArrayAttr;
  StringAttr sym = {};

  bool forceable = !!firrtl::detail::getForceableResultType(true, type);
  auto result = builder.create<WireOp>(type, id, NameKindEnum::InterestingName,
                                       annotations, sym, forceable);
  return moduleContext.addSymbolEntry(id, result.getResult(),
                                      startTok.getLoc());
}

/// register    ::= 'reg' id ':' type exp ('with' ':' reset_block)? info?
///
/// reset_block ::= INDENT simple_reset info? NEWLINE DEDENT
///             ::= '(' simple_reset ')'
///
/// simple_reset ::= simple_reset0
///              ::= '(' simple_reset0 ')'
///
/// simple_reset0:  'reset' '=>' '(' exp exp ')'
///
ParseResult FIRStmtParser::parseRegister(unsigned regIndent) {
  auto startTok = consumeToken(FIRToken::kw_reg);

  // If this was actually the start of a connect or something else handle
  // that.
  if (auto isExpr = parseExpWithLeadingKeyword(startTok))
    return *isExpr;

  StringRef id;
  FIRRTLType type;
  Value clock;

  // TODO(firrtl spec): info? should come after the clock expression before
  // the 'with'.
  if (parseId(id, "expected reg name") ||
      parseToken(FIRToken::colon, "expected ':' in reg") ||
      parseType(type, "expected reg type") ||
      parseExp(clock, "expected expression for register clock"))
    return failure();

  if (!type_isa<FIRRTLBaseType>(type))
    return emitError(startTok.getLoc(), "register must have base type");

  // Parse the 'with' specifier if present.
  Value resetSignal, resetValue;
  if (consumeIf(FIRToken::kw_with)) {
    if (parseToken(FIRToken::colon, "expected ':' in reg"))
      return failure();

    // TODO(firrtl spec): Simplify the grammar for register reset logic.
    // Why allow multiple ambiguous parentheses?  Why rely on indentation at
    // all?

    // This implements what the examples have in practice.
    bool hasExtraLParen = consumeIf(FIRToken::l_paren);

    auto indent = getIndentation();
    if (!indent || *indent <= regIndent)
      if (!hasExtraLParen)
        return emitError("expected indented reset specifier in reg"), failure();

    if (parseToken(FIRToken::kw_reset, "expected 'reset' in reg") ||
        parseToken(FIRToken::equal_greater, "expected => in reset specifier") ||
        parseToken(FIRToken::l_paren, "expected '(' in reset specifier") ||
        parseExp(resetSignal, "expected expression for reset signal"))
      return failure();

    // The Scala implementation of FIRRTL represents registers without resets
    // as a self referential register... and the pretty printer doesn't print
    // the right form. Recognize that this is happening and treat it as a
    // register without a reset for compatibility.
    // TODO(firrtl scala impl): pretty print registers without resets right.
    if (getTokenSpelling() == id) {
      consumeToken();
      if (parseToken(FIRToken::r_paren, "expected ')' in reset specifier"))
        return failure();
      resetSignal = Value();
    } else {
      if (parseExp(resetValue, "expected expression for reset value") ||
          parseToken(FIRToken::r_paren, "expected ')' in reset specifier"))
        return failure();
    }

    if (hasExtraLParen &&
        parseToken(FIRToken::r_paren, "expected ')' in reset specifier"))
      return failure();
  }

  // Finally, handle the last info if present, providing location info for the
  // clock expression.
  if (parseOptionalInfo())
    return failure();

  locationProcessor.setLoc(startTok.getLoc());

  ArrayAttr annotations = getConstants().emptyArrayAttr;
  Value result;
  StringAttr sym = {};
  bool forceable = !!firrtl::detail::getForceableResultType(true, type);
  if (resetSignal)
    result = builder
                 .create<RegResetOp>(type, clock, resetSignal, resetValue, id,
                                     NameKindEnum::InterestingName, annotations,
                                     sym, forceable)
                 .getResult();
  else
    result = builder
                 .create<RegOp>(type, clock, id, NameKindEnum::InterestingName,
                                annotations, sym, forceable)
                 .getResult();
  return moduleContext.addSymbolEntry(id, result, startTok.getLoc());
}

/// registerWithReset ::= 'regreset' id ':' type exp exp exp
///
/// This syntax is only supported in FIRRTL versions >= 3.0.0.  Because this
/// syntax is only valid for >= 3.0.0, there is no need to check if the leading
/// "regreset" is part of an expression with a leading keyword.
ParseResult FIRStmtParser::parseRegisterWithReset() {
  auto startTok = consumeToken(FIRToken::kw_regreset);

  StringRef id;
  FIRRTLType type;
  Value clock, resetSignal, resetValue;

  if (parseId(id, "expected reg name") ||
      parseToken(FIRToken::colon, "expected ':' in reg") ||
      parseType(type, "expected reg type") ||
      parseExp(clock, "expected expression for register clock") ||
      parseExp(resetSignal, "expected expression for register reset") ||
      parseExp(resetValue, "expected expression for register reset value") ||
      parseOptionalInfo())
    return failure();

  if (!type_isa<FIRRTLBaseType>(type))
    return emitError(startTok.getLoc(), "register must have base type");

  locationProcessor.setLoc(startTok.getLoc());

  bool forceable = !!firrtl::detail::getForceableResultType(true, type);
  auto result = builder
                    .create<RegResetOp>(type, clock, resetSignal, resetValue,
                                        id, NameKindEnum::InterestingName,
                                        getConstants().emptyArrayAttr,
                                        StringAttr{}, forceable)
                    .getResult();

  return moduleContext.addSymbolEntry(id, result, startTok.getLoc());
}

//===----------------------------------------------------------------------===//
// FIRCircuitParser
//===----------------------------------------------------------------------===//

namespace {
/// This class implements the outer level of the parser, including things
/// like circuit and module.
struct FIRCircuitParser : public FIRParser {
  explicit FIRCircuitParser(SharedParserConstants &state, FIRLexer &lexer,
                            ModuleOp mlirModule, FIRVersion &version)
      : FIRParser(state, lexer, version), mlirModule(mlirModule) {}

  ParseResult
  parseCircuit(SmallVectorImpl<const llvm::MemoryBuffer *> &annotationsBuf,
               SmallVectorImpl<const llvm::MemoryBuffer *> &omirBuf,
               mlir::TimingScope &ts);

private:
  /// Extract Annotations from a JSON-encoded Annotation array string and add
  /// them to a vector of attributes.
  ParseResult importAnnotationsRaw(SMLoc loc, StringRef annotationsStr,
                                   SmallVectorImpl<Attribute> &attrs);
  /// Generate OMIR-derived annotations.  Report errors if the OMIR is malformed
  /// in any way.  This also performs scattering of the OMIR to introduce
  /// tracking annotations in the circuit.
  ParseResult importOMIR(CircuitOp circuit, SMLoc loc, StringRef annotationStr,
                         SmallVectorImpl<Attribute> &attrs);

  ParseResult parseToplevelDefinition(CircuitOp circuit, unsigned indent);

  ParseResult parseClass(CircuitOp circuit, unsigned indent);
  ParseResult parseExtModule(CircuitOp circuit, unsigned indent);
  ParseResult parseIntModule(CircuitOp circuit, unsigned indent);
  ParseResult parseModule(CircuitOp circuit, unsigned indent);

  ParseResult parsePortList(SmallVectorImpl<PortInfo> &resultPorts,
                            SmallVectorImpl<SMLoc> &resultPortLocs,
                            unsigned indent);
  ParseResult parseParameterList(ArrayAttr &resultParameters);
  ParseResult parseParameter(StringAttr &resultName, TypedAttr &resultValue,
                             SMLoc &resultLoc);
  ParseResult parseRefList(ArrayRef<PortInfo> portList,
                           ArrayAttr &internalPathResults);

  ParseResult skipToModuleEnd(unsigned indent);

  ParseResult parseTypeDecl();

  ParseResult parseGroupDecl(CircuitOp circuit);

  struct DeferredModuleToParse {
    FModuleLike moduleOp;
    SmallVector<SMLoc> portLocs;
    FIRLexerCursor lexerCursor;
    unsigned indent;
  };

  ParseResult parseModuleBody(DeferredModuleToParse &deferredModule);

  SmallVector<DeferredModuleToParse, 0> deferredModules;
  ModuleOp mlirModule;
};

} // end anonymous namespace
ParseResult
FIRCircuitParser::importAnnotationsRaw(SMLoc loc, StringRef annotationsStr,
                                       SmallVectorImpl<Attribute> &attrs) {

  auto annotations = json::parse(annotationsStr);
  if (auto err = annotations.takeError()) {
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {
      auto diag = emitError(loc, "Failed to parse JSON Annotations");
      diag.attachNote() << a.message();
    });
    return failure();
  }

  json::Path::Root root;
  llvm::StringMap<ArrayAttr> thisAnnotationMap;
  if (!fromJSONRaw(annotations.get(), attrs, root, getContext())) {
    auto diag = emitError(loc, "Invalid/unsupported annotation format");
    std::string jsonErrorMessage =
        "See inline comments for problem area in JSON:\n";
    llvm::raw_string_ostream s(jsonErrorMessage);
    root.printErrorContext(annotations.get(), s);
    diag.attachNote() << jsonErrorMessage;
    return failure();
  }

  return success();
}

ParseResult FIRCircuitParser::importOMIR(CircuitOp circuit, SMLoc loc,
                                         StringRef annotationsStr,
                                         SmallVectorImpl<Attribute> &annos) {

  auto annotations = json::parse(annotationsStr);
  if (auto err = annotations.takeError()) {
    handleAllErrors(std::move(err), [&](const json::ParseError &a) {
      auto diag = emitError(loc, "Failed to parse OMIR file");
      diag.attachNote() << a.message();
    });
    return failure();
  }

  json::Path::Root root;
  if (!fromOMIRJSON(annotations.get(), annos, root, circuit.getContext())) {
    auto diag = emitError(loc, "Invalid/unsupported OMIR format");
    std::string jsonErrorMessage =
        "See inline comments for problem area in JSON:\n";
    llvm::raw_string_ostream s(jsonErrorMessage);
    root.printErrorContext(annotations.get(), s);
    diag.attachNote() << jsonErrorMessage;
    return failure();
  }

  return success();
}

/// portlist ::= port*
/// port     ::= dir id ':' type info? NEWLINE
/// dir      ::= 'input' | 'output'
ParseResult
FIRCircuitParser::parsePortList(SmallVectorImpl<PortInfo> &resultPorts,
                                SmallVectorImpl<SMLoc> &resultPortLocs,
                                unsigned indent) {
  // Parse any ports.
  while (getToken().isAny(FIRToken::kw_input, FIRToken::kw_output) &&
         // Must be nested under the module.
         getIndentation() > indent) {

    // We need one token lookahead to resolve the ambiguity between:
    // output foo             ; port
    // output <= input        ; identifier expression
    // output.thing <= input  ; identifier expression
    auto backtrackState = getLexer().getCursor();

    bool isOutput = getToken().is(FIRToken::kw_output);
    consumeToken();

    // If we have something that isn't a keyword then this must be an
    // identifier, not an input/output marker.
    if (!getToken().isAny(FIRToken::identifier, FIRToken::literal_identifier) &&
        !getToken().isKeyword()) {
      backtrackState.restore(getLexer());
      break;
    }

    StringAttr name;
    FIRRTLType type;
    LocWithInfo info(getToken().getLoc(), this);
    if (parseId(name, "expected port name") ||
        parseToken(FIRToken::colon, "expected ':' in port definition") ||
        parseType(type, "expected a type in port declaration") ||
        info.parseOptionalInfo())
      return failure();

    StringAttr innerSym = {};
    resultPorts.push_back(
        {name, type, direction::get(isOutput), innerSym, info.getLoc()});
    resultPortLocs.push_back(info.getFIRLoc());
  }

  // Helper for the temporary check rejecting input-oriented refs.
  std::function<bool(Type, bool)> hasInputRef = [&](Type type,
                                                    bool output) -> bool {
    auto ftype = type_dyn_cast<FIRRTLType>(type);
    if (!ftype || !ftype.containsReference())
      return false;
    return TypeSwitch<FIRRTLType, bool>(ftype)
        .Case<RefType>([&](auto reftype) { return !output; })
        .Case<OpenVectorType>([&](OpenVectorType ovt) {
          return hasInputRef(ovt.getElementType(), output);
        })
        .Case<OpenBundleType>([&](OpenBundleType obt) {
          for (auto field : obt.getElements())
            if (hasInputRef(field.type, field.isFlip ^ output))
              return true;
          return false;
        });
  };

  // Check for port name collisions.
  SmallDenseMap<Attribute, SMLoc> portIds;
  for (auto portAndLoc : llvm::zip(resultPorts, resultPortLocs)) {
    PortInfo &port = std::get<0>(portAndLoc);
    // See #4812 and look through the reference input test collection
    // and ensure they work before allowing them from user input.
    if (hasInputRef(port.type, port.isOutput()))
      return emitError(std::get<1>(portAndLoc),
                       "input probes not yet supported");
    auto &entry = portIds[port.name];
    if (!entry.isValid()) {
      entry = std::get<1>(portAndLoc);
      continue;
    }

    emitError(std::get<1>(portAndLoc),
              "redefinition of name '" + port.getName() + "'")
            .attachNote(translateLocation(entry))
        << "previous definition here";
    return failure();
  }

  return success();
}

/// ref-list ::= ref*
/// ref ::= 'ref' static_reference 'is' StringLit NEWLIN
ParseResult FIRCircuitParser::parseRefList(ArrayRef<PortInfo> portList,
                                           ArrayAttr &internalPathsResult) {
  struct RefStatementInfo {
    StringAttr refName;
    StringAttr resolvedPath;
    SMLoc loc;
  };

  SmallVector<RefStatementInfo> refStatements;
  SmallPtrSet<StringAttr, 8> seenNames;
  SmallPtrSet<StringAttr, 8> seenRefs;

  // Parse the ref statements.
  while (consumeIf(FIRToken::kw_ref)) {
    auto loc = getToken().getLoc();
    // ref x is "a.b.c"
    // Support "ref x.y is " once aggregate-of-ref supported.
    StringAttr refName, resolved;
    if (parseId(refName, "expected ref name"))
      return failure();
    if (consumeIf(FIRToken::period) || consumeIf(FIRToken::l_square))
      return emitError(
          loc, "ref statements for aggregate elements not yet supported");
    if (parseToken(FIRToken::kw_is, "expected 'is' in ref statement"))
      return failure();

    if (!seenRefs.insert(refName).second)
      return emitError(loc, "duplicate ref statement for '" + refName.strref() +
                                "'");

    auto kind = getToken().getKind();
    if (kind != FIRToken::string)
      return emitError(loc, "expected string in ref statement");
    resolved = StringAttr::get(getContext(), getToken().getStringValue());
    consumeToken(FIRToken::string);

    refStatements.push_back(RefStatementInfo{refName, resolved, loc});
  }

  // Build paths array.  One entry for each ref-type port.
  SmallVector<Attribute> internalPaths;
  auto refPorts = llvm::make_filter_range(
      portList, [&](auto &port) { return type_isa<RefType>(port.type); });
  llvm::SmallBitVector usedRefs(refStatements.size());
  for (auto &port : refPorts) {
    // Reject input reftype ports on extmodule's per spec,
    // as well as on intmodule's which is not mentioned in spec.
    if (!port.isOutput())
      return mlir::emitError(
          port.loc,
          "references in ports must be output on extmodule and intmodule");
    auto *refStmtIt = llvm::find_if(
        refStatements, [&](const auto &r) { return r.refName == port.name; });
    // Error if ref statements are present but none found for this port.
    if (refStmtIt == refStatements.end()) {
      if (!refStatements.empty())
        return mlir::emitError(port.loc, "no ref statement found for ref port ")
            .append(port.name);
      continue;
    }

    usedRefs.set(std::distance(refStatements.begin(), refStmtIt));
    internalPaths.push_back(refStmtIt->resolvedPath);
  }

  if (!refStatements.empty() && internalPaths.size() != refStatements.size()) {
    assert(internalPaths.size() < refStatements.size());
    assert(!usedRefs.all());
    auto idx = usedRefs.find_first_unset();
    assert(idx != -1);
    return emitError(refStatements[idx].loc, "unused ref statement");
  }

  internalPathsResult = ArrayAttr::get(getContext(), internalPaths);
  return success();
}

/// We're going to defer parsing this module, so just skip tokens until we
/// get to the next module or the end of the file.
ParseResult FIRCircuitParser::skipToModuleEnd(unsigned indent) {
  while (true) {
    switch (getToken().getKind()) {

    // End of file or invalid token will be handled by outer level.
    case FIRToken::eof:
    case FIRToken::error:
      return success();

    // If we got to the next top-level declaration, then we're done.
    case FIRToken::kw_class:
    case FIRToken::kw_declgroup:
    case FIRToken::kw_extmodule:
    case FIRToken::kw_intmodule:
    case FIRToken::kw_module:
    case FIRToken::kw_type:
      // All module declarations should have the same indentation
      // level. Use this fact to differentiate between module
      // declarations and usages of "module" as identifiers.
      if (getIndentation() == indent)
        return success();
      [[fallthrough]];
    default:
      consumeToken();
      break;
    }
  }
}

/// parameter ::= 'parameter' id '=' intLit NEWLINE
/// parameter ::= 'parameter' id '=' StringLit NEWLINE
/// parameter ::= 'parameter' id '=' floatingpoint NEWLINE
/// parameter ::= 'parameter' id '=' RawString NEWLINE
ParseResult FIRCircuitParser::parseParameter(StringAttr &resultName,
                                             TypedAttr &resultValue,
                                             SMLoc &resultLoc) {
  mlir::Builder builder(getContext());

  consumeToken(FIRToken::kw_parameter);
  auto loc = getToken().getLoc();

  StringRef name;
  if (parseId(name, "expected parameter name") ||
      parseToken(FIRToken::equal, "expected '=' in parameter"))
    return failure();

  TypedAttr value;
  switch (getToken().getKind()) {
  default:
    return emitError("expected parameter value"), failure();
  case FIRToken::integer:
  case FIRToken::signed_integer: {
    APInt result;
    if (parseIntLit(result, "invalid integer parameter"))
      return failure();

    // If the integer parameter is less than 32-bits, sign extend this to a
    // 32-bit value.  This needs to eventually emit as a 32-bit value in
    // Verilog and we want to get the size correct immediately.
    if (result.getBitWidth() < 32)
      result = result.sext(32);

    value = builder.getIntegerAttr(
        builder.getIntegerType(result.getBitWidth(), result.isSignBitSet()),
        result);
    break;
  }
  case FIRToken::string: {
    // Drop the double quotes and unescape.
    value = builder.getStringAttr(getToken().getStringValue());
    consumeToken(FIRToken::string);
    break;
  }
  case FIRToken::raw_string: {
    // Drop the single quotes and unescape the ones inside.
    value = builder.getStringAttr(getToken().getRawStringValue());
    consumeToken(FIRToken::raw_string);
    break;
  }
  case FIRToken::floatingpoint:
    double v;
    if (!llvm::to_float(getTokenSpelling(), v))
      return emitError("invalid float parameter syntax"), failure();

    value = builder.getF64FloatAttr(v);
    consumeToken(FIRToken::floatingpoint);
    break;
  }

  resultName = builder.getStringAttr(name);
  resultValue = value;
  resultLoc = loc;
  return success();
}

/// parameter-list ::= parameter*
ParseResult FIRCircuitParser::parseParameterList(ArrayAttr &resultParameters) {
  SmallVector<Attribute, 8> parameters;
  SmallPtrSet<StringAttr, 8> seen;
  while (getToken().is(FIRToken::kw_parameter)) {
    StringAttr name;
    TypedAttr value;
    SMLoc loc;
    if (parseParameter(name, value, loc))
      return failure();
    if (!seen.insert(name).second)
      return emitError(loc,
                       "redefinition of parameter '" + name.getValue() + "'");
    parameters.push_back(ParamDeclAttr::get(name, value));
  }
  resultParameters = ArrayAttr::get(getContext(), parameters);
  return success();
}

/// class ::= 'class' id ':' info? INDENT portlist simple_stmt_block DEDENT
ParseResult FIRCircuitParser::parseClass(CircuitOp circuit, unsigned indent) {
  StringAttr name;
  SmallVector<PortInfo, 8> portList;
  SmallVector<SMLoc> portLocs;
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_class);
  if (parseId(name, "expected class name") ||
      parseToken(FIRToken::colon, "expected ':' in class definition") ||
      info.parseOptionalInfo() || parsePortList(portList, portLocs, indent))
    return failure();

  if (name == circuit.getName())
    return mlir::emitError(info.getLoc(),
                           "class cannot be the top of a circuit");

  for (auto &portInfo : portList)
    if (!isa<PropertyType>(portInfo.type))
      return mlir::emitError(portInfo.loc,
                             "ports on classes must be properties");

  // build it
  auto builder = circuit.getBodyBuilder();
  auto classOp = builder.create<ClassOp>(info.getLoc(), name, portList);
  deferredModules.emplace_back(
      DeferredModuleToParse{classOp, portLocs, getLexer().getCursor(), indent});

  return skipToModuleEnd(indent);
}

/// extmodule ::=
///        'extmodule' id ':' info?
///        INDENT portlist defname? parameter-list ref-list DEDENT
/// defname   ::= 'defname' '=' id NEWLINE
ParseResult FIRCircuitParser::parseExtModule(CircuitOp circuit,
                                             unsigned indent) {
  StringAttr name;
  SmallVector<PortInfo, 8> portList;
  SmallVector<SMLoc> portLocs;
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_extmodule);
  if (parseId(name, "expected extmodule name") ||
      parseToken(FIRToken::colon, "expected ':' in extmodule definition") ||
      info.parseOptionalInfo() || parsePortList(portList, portLocs, indent))
    return failure();

  StringRef defName;
  if (consumeIf(FIRToken::kw_defname)) {
    if (parseToken(FIRToken::equal, "expected '=' in defname") ||
        parseId(defName, "expected defname name"))
      return failure();
  }

  ArrayAttr parameters;
  ArrayAttr internalPaths;
  if (parseParameterList(parameters) || parseRefList(portList, internalPaths))
    return failure();

  auto builder = circuit.getBodyBuilder();
  auto convention = getConstants().options.scalarizeExtModules
                        ? Convention::Scalarized
                        : Convention::Internal;
  auto conventionAttr = ConventionAttr::get(getContext(), convention);
  auto annotations = ArrayAttr::get(getContext(), {});
  builder.create<FExtModuleOp>(info.getLoc(), name, conventionAttr, portList,
                               defName, annotations, parameters, internalPaths);
  return success();
}

/// intmodule ::=
///        'intmodule' id ':' info?
///        INDENT portlist intname parameter-list ref-list DEDENT
/// intname   ::= 'intrinsic' '=' id NEWLINE
ParseResult FIRCircuitParser::parseIntModule(CircuitOp circuit,
                                             unsigned indent) {
  StringAttr name;
  SmallVector<PortInfo, 8> portList;
  SmallVector<SMLoc> portLocs;
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_intmodule);
  if (parseId(name, "expected intmodule name") ||
      parseToken(FIRToken::colon, "expected ':' in intmodule definition") ||
      info.parseOptionalInfo() || parsePortList(portList, portLocs, indent))
    return failure();

  StringRef intName;
  if (consumeIf(FIRToken::kw_intrinsic)) {
    if (parseToken(FIRToken::equal, "expected '=' in intrinsic") ||
        parseId(intName, "expected intrinsic name"))
      return failure();
  }

  ArrayAttr parameters;
  ArrayAttr internalPaths;
  if (parseParameterList(parameters) || parseRefList(portList, internalPaths))
    return failure();

  ArrayAttr annotations = getConstants().emptyArrayAttr;
  auto builder = circuit.getBodyBuilder();
  builder.create<FIntModuleOp>(info.getLoc(), name, portList, intName,
                               annotations, parameters, internalPaths);
  return success();
}

/// module ::= 'module' id ':' info? INDENT portlist simple_stmt_block DEDENT
ParseResult FIRCircuitParser::parseModule(CircuitOp circuit, unsigned indent) {
  StringAttr name;
  SmallVector<PortInfo, 8> portList;
  SmallVector<SMLoc> portLocs;
  LocWithInfo info(getToken().getLoc(), this);
  consumeToken(FIRToken::kw_module);
  if (parseId(name, "expected module name") ||
      parseToken(FIRToken::colon, "expected ':' in module definition") ||
      info.parseOptionalInfo() || parsePortList(portList, portLocs, indent))
    return failure();

  auto circuitName = circuit.getName();
  auto isMainModule = (name == circuitName);
  ArrayAttr annotations = getConstants().emptyArrayAttr;
  auto convention = Convention::Internal;
  if (isMainModule && getConstants().options.scalarizeTopModule)
    convention = Convention::Scalarized;
  auto conventionAttr = ConventionAttr::get(getContext(), convention);
  auto builder = circuit.getBodyBuilder();
  auto moduleOp = builder.create<FModuleOp>(info.getLoc(), name, conventionAttr,
                                            portList, annotations);
  auto visibility = isMainModule ? SymbolTable::Visibility::Public
                                 : SymbolTable::Visibility::Private;
  SymbolTable::setSymbolVisibility(moduleOp, visibility);

  // Parse the body of this module after all prototypes have been parsed. This
  // allows us to handle forward references correctly.
  deferredModules.emplace_back(DeferredModuleToParse{
      moduleOp, portLocs, getLexer().getCursor(), indent});

  if (skipToModuleEnd(indent))
    return failure();
  return success();
}

ParseResult FIRCircuitParser::parseToplevelDefinition(CircuitOp circuit,
                                                      unsigned indent) {
  switch (getToken().getKind()) {
  case FIRToken::kw_class:
    return parseClass(circuit, indent);
  case FIRToken::kw_declgroup:
    if (FIRVersion::compare(version, FIRVersion({3, 1, 0})) < 0)
      return emitError()
             << "unexpected token: optional groups are a FIRRTL 3.1.0+ "
                "feature, but the specified FIRRTL version was "
             << version;
    return parseGroupDecl(circuit);
  case FIRToken::kw_extmodule:
    return parseExtModule(circuit, indent);
  case FIRToken::kw_intmodule:
    return parseIntModule(circuit, indent);
  case FIRToken::kw_module:
    return parseModule(circuit, indent);
  case FIRToken::kw_type:
    return parseTypeDecl();
  default:
    return emitError(getToken().getLoc(), "unknown toplevel definition");
  }
}

// Parse a type declaration.
ParseResult FIRCircuitParser::parseTypeDecl() {
  StringRef id;
  FIRRTLType type;
  consumeToken();
  auto loc = getToken().getLoc();

  if (getToken().isKeyword())
    return emitError(loc) << "cannot use keyword '" << getToken().getSpelling()
                          << "' for type alias name";

  if (parseId(id, "expected type name") ||
      parseToken(FIRToken::equal, "expected '=' in type decl") ||
      parseType(type, "expected a type"))
    return failure();
  auto name = StringAttr::get(type.getContext(), id);
  // Create type alias only for base types. Otherwise just pass through the
  // type.
  if (auto base = type_dyn_cast<FIRRTLBaseType>(type))
    type = BaseTypeAliasType::get(name, base);
  else
    emitWarning(loc)
        << "type alias for non-base type " << type
        << " is currently not supported. Type alias is stripped immediately";

  if (!getConstants().aliasMap.insert({id, type}).second)
    return emitError(loc) << "type alias `" << name.getValue()
                          << "` is already defined";
  return success();
}

// Parse a group declaration.
ParseResult FIRCircuitParser::parseGroupDecl(CircuitOp circuit) {
  auto baseIndent = getIndentation();

  // A stack of all groups that are possibly parents of the current group.
  SmallVector<std::pair<std::optional<unsigned>, GroupDeclOp>> groupStack;

  // Parse a single group and add it to the groupStack.
  auto parseOne = [&](Block *block) -> ParseResult {
    auto indent = getIndentation();
    StringRef id, convention;
    LocWithInfo info(getToken().getLoc(), this);
    consumeToken();
    if (parseId(id, "expected group name") || parseGetSpelling(convention))
      return failure();
    auto groupConvention = symbolizeGroupConvention(convention);
    if (!groupConvention) {
      emitError() << "unknown convention '" << convention
                  << "' (did you misspell it?)";
      return failure();
    }
    consumeToken();
    if (parseToken(FIRToken::colon, "expected ':' after group definition") ||
        info.parseOptionalInfo())
      return failure();
    auto builder = OpBuilder::atBlockEnd(block);
    // Create the group declaration and give it an empty block.
    auto groupDeclOp =
        builder.create<GroupDeclOp>(info.getLoc(), id, *groupConvention);
    groupDeclOp->getRegion(0).push_back(new Block());
    groupStack.push_back({indent, groupDeclOp});
    return success();
  };

  if (parseOne(circuit.getBodyBlock()))
    return failure();

  // Parse any nested groups.
  while (getIndentation() > baseIndent) {
    switch (getToken().getKind()) {
    case FIRToken::kw_declgroup: {
      // Pop nested groups off the stack until we find out what group to insert
      // this into.
      while (groupStack.back().first >= getIndentation())
        groupStack.pop_back();
      auto parentGroup = groupStack.back().second;
      if (parseOne(&parentGroup.getBody().front()))
        return failure();
      break;
    }
    default:
      return emitError("expected 'declgroup'"), failure();
    }
  }

  return success();
}

// Parse the body of this module.
ParseResult
FIRCircuitParser::parseModuleBody(DeferredModuleToParse &deferredModule) {
  FModuleLike moduleOp = deferredModule.moduleOp;
  auto &body = moduleOp->getRegion(0).front();
  auto &portLocs = deferredModule.portLocs;

  // We parse the body of this module with its own lexer, enabling parallel
  // parsing with the rest of the other module bodies.
  FIRLexer moduleBodyLexer(getLexer().getSourceMgr(), getContext());

  // Reset the parser/lexer state back to right after the port list.
  deferredModule.lexerCursor.restore(moduleBodyLexer);

  FIRModuleContext moduleContext(getConstants(), moduleBodyLexer, version);

  // Install all of the ports into the symbol table, associated with their
  // block arguments.
  auto portList = moduleOp.getPorts();
  auto portArgs = body.getArguments();
  for (auto tuple : llvm::zip(portList, portLocs, portArgs)) {
    PortInfo &port = std::get<0>(tuple);
    llvm::SMLoc loc = std::get<1>(tuple);
    BlockArgument portArg = std::get<2>(tuple);
    assert(!port.sym);
    if (moduleContext.addSymbolEntry(port.getName(), portArg, loc))
      return failure();
  }

  ModuleNamespace modNameSpace(moduleOp);
  FIRStmtParser stmtParser(body, moduleContext, modNameSpace, version);

  // Parse the moduleBlock.
  auto result = stmtParser.parseSimpleStmtBlock(deferredModule.indent);
  if (failed(result))
    return result;

  // Convert print-encoded verifications after parsing.

  // It is dangerous to modify IR in the walk, so accumulate printFOp to
  // buffer.
  SmallVector<PrintFOp> buffer;
  deferredModule.moduleOp.walk(
      [&buffer](PrintFOp printFOp) { buffer.push_back(printFOp); });

  for (auto printFOp : buffer) {
    auto result = circt::firrtl::foldWhenEncodedVerifOp(printFOp);
    if (failed(result))
      return result;
  }

  // Demote any forceable operations that aren't being forced.
  deferredModule.moduleOp.walk([](Forceable fop) {
    if (fop.isForceable() && fop.getDataRef().use_empty())
      firrtl::detail::replaceWithNewForceability(fop, false);
  });

  return success();
}

/// file ::= circuit
/// versionHeader ::= 'FIRRTL' 'version' versionLit NEWLINE
/// circuit ::= versionHeader? 'circuit' id ':' info? INDENT module* DEDENT EOF
///
/// If non-null, annotationsBuf is a memory buffer containing JSON annotations.
/// If non-null, omirBufs is a vector of memory buffers containing SiFive Object
/// Model IR (which is JSON).
///
ParseResult FIRCircuitParser::parseCircuit(
    SmallVectorImpl<const llvm::MemoryBuffer *> &annotationsBufs,
    SmallVectorImpl<const llvm::MemoryBuffer *> &omirBufs,
    mlir::TimingScope &ts) {

  auto indent = getIndentation();
  if (consumeIf(FIRToken::kw_FIRRTL)) {
    if (!indent.has_value())
      return emitError("'FIRRTL' must be first token on its line"), failure();
    if (parseToken(FIRToken::kw_version, "expected version after 'FIRRTL'") ||
        parseVersionLit("expected version literal"))
      return failure();
    indent = getIndentation();
  }

  if (!indent.has_value())
    return emitError("'circuit' must be first token on its line"), failure();
  unsigned circuitIndent = *indent;

  LocWithInfo info(getToken().getLoc(), this);
  StringAttr name;
  SMLoc inlineAnnotationsLoc;
  StringRef inlineAnnotations;

  // A file must contain a top level `circuit` definition.
  if (parseToken(FIRToken::kw_circuit,
                 "expected a top-level 'circuit' definition") ||
      parseId(name, "expected circuit name") ||
      parseToken(FIRToken::colon, "expected ':' in circuit definition") ||
      parseOptionalAnnotations(inlineAnnotationsLoc, inlineAnnotations) ||
      info.parseOptionalInfo())
    return failure();

  // Create the top-level circuit op in the MLIR module.
  OpBuilder b(mlirModule.getBodyRegion());
  auto circuit = b.create<CircuitOp>(info.getLoc(), name);

  // A timer to get execution time of annotation parsing.
  auto parseAnnotationTimer = ts.nest("Parse annotations");

  // Deal with any inline annotations, if they exist.  These are processed
  // first to place any annotations from an annotation file *after* the inline
  // annotations.  While arbitrary, this makes the annotation file have
  // "append" semantics.
  SmallVector<Attribute> annos;
  if (!inlineAnnotations.empty())
    if (importAnnotationsRaw(inlineAnnotationsLoc, inlineAnnotations, annos))
      return failure();

  // Deal with the annotation file if one was specified
  for (auto *annotationsBuf : annotationsBufs)
    if (importAnnotationsRaw(info.getFIRLoc(), annotationsBuf->getBuffer(),
                             annos))
      return failure();

  parseAnnotationTimer.stop();
  auto parseOMIRTimer = ts.nest("Parse OMIR");

  // Process OMIR files as annotations with a class of
  // "freechips.rocketchip.objectmodel.OMNode"
  for (auto *omirBuf : omirBufs)
    if (importOMIR(circuit, info.getFIRLoc(), omirBuf->getBuffer(), annos))
      return failure();

  parseOMIRTimer.stop();

  // Get annotations that are supposed to be specially handled by the
  // LowerAnnotations pass.
  if (!annos.empty())
    circuit->setAttr(rawAnnotations, b.getArrayAttr(annos));

  // A timer to get execution time of module parsing.
  auto parseTimer = ts.nest("Parse modules");
  deferredModules.reserve(16);

  // Parse any contained modules.
  while (true) {
    switch (getToken().getKind()) {
    // If we got to the end of the file, then we're done.
    case FIRToken::eof:
      goto DoneParsing;

    // If we got an error token, then the lexer already emitted an error,
    // just stop.  We could introduce error recovery if there was demand for
    // it.
    case FIRToken::error:
      return failure();

    default:
      emitError("unexpected token in circuit");
      return failure();

    case FIRToken::kw_class:
    case FIRToken::kw_declgroup:
    case FIRToken::kw_extmodule:
    case FIRToken::kw_intmodule:
    case FIRToken::kw_module:
    case FIRToken::kw_type: {
      auto indent = getIndentation();
      if (!indent.has_value())
        return emitError("'module' must be first token on its line"), failure();
      unsigned definitionIndent = *indent;

      if (definitionIndent <= circuitIndent)
        return emitError("module should be indented more"), failure();

      if (parseToplevelDefinition(circuit, definitionIndent))
        return failure();
      break;
    }
    }
  }

  // After the outline of the file has been parsed, we can go ahead and parse
  // all the bodies.  This allows us to resolve forward-referenced modules and
  // makes it possible to parse their bodies in parallel.
DoneParsing:
  // Each of the modules may translate source locations, and doing so touches
  // the SourceMgr to build a line number cache.  This isn't thread safe, so we
  // proactively touch it to make sure that it is always already created.
  (void)getLexer().translateLocation(info.getFIRLoc());

  // Next, parse all the module bodies.
  auto anyFailed = mlir::failableParallelForEachN(
      getContext(), 0, deferredModules.size(), [&](size_t index) {
        if (parseModuleBody(deferredModules[index]))
          return failure();
        return success();
      });
  if (failed(anyFailed))
    return failure();

  auto main = circuit.getMainModule();
  if (!main) {
    // Give more specific error if no modules defined at all
    if (circuit.getOps<FModuleLike>().empty()) {
      return mlir::emitError(circuit.getLoc())
             << "no modules found, circuit must contain one or more modules";
    }
    if (auto *notModule = circuit.lookupSymbol(circuit.getName())) {
      return notModule->emitOpError()
             << "cannot have the same name as the circuit";
    }
    return mlir::emitError(circuit.getLoc())
           << "no main module found, circuit '" << circuit.getName()
           << "' must contain a module named '" << circuit.getName() << "'";
  }

  // If the circuit has an entry point that is not an external module, set the
  // visibility of all non-main modules to private.
  if (auto mainMod = dyn_cast<FModuleOp>(*main)) {
    for (auto mod : circuit.getOps<FModuleLike>()) {
      if (mod != main)
        SymbolTable::setSymbolVisibility(mod, SymbolTable::Visibility::Private);
    }
    // Reject if main module has input ref-type ports.
    // This should be checked in verifier for all public FModuleLike's but
    // they're used internally so check this here.
    for (auto &pi : mainMod.getPorts()) {
      if (!pi.isOutput() && type_isa<RefType>(pi.type))
        return mlir::emitError(pi.loc)
               << "main module may not contain input references";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

// Parse the specified .fir file into the specified MLIR context.
mlir::OwningOpRef<mlir::ModuleOp>
circt::firrtl::importFIRFile(SourceMgr &sourceMgr, MLIRContext *context,
                             mlir::TimingScope &ts, FIRParserOptions options) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  SmallVector<const llvm::MemoryBuffer *> annotationsBufs;
  unsigned fileID = 1;
  for (unsigned e = options.numAnnotationFiles + 1; fileID < e; ++fileID)
    annotationsBufs.push_back(
        sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID() + fileID));

  SmallVector<const llvm::MemoryBuffer *> omirBufs;
  for (unsigned e = sourceMgr.getNumBuffers(); fileID < e; ++fileID)
    omirBufs.push_back(
        sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID() + fileID));

  context->loadDialect<CHIRRTLDialect>();
  context->loadDialect<FIRRTLDialect, hw::HWDialect>();

  // This is the result module we are parsing into.
  mlir::OwningOpRef<mlir::ModuleOp> module(ModuleOp::create(
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0)));
  SharedParserConstants state(context, options);
  FIRLexer lexer(sourceMgr, context);
  FIRVersion version = FIRVersion::defaultFIRVersion();
  if (FIRCircuitParser(state, lexer, *module, version)
          .parseCircuit(annotationsBufs, omirBufs, ts))
    return nullptr;

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto circuitVerificationTimer = ts.nest("Verify circuit");
  if (failed(verify(*module)))
    return {};

  return module;
}

void circt::firrtl::registerFromFIRFileTranslation() {
  static mlir::TranslateToMLIRRegistration fromFIR(
      "import-firrtl", "import .fir",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *context) {
        mlir::TimingScope ts;
        return importFIRFile(sourceMgr, context, ts);
      });
}
