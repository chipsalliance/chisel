// See LICENSE for license details.

grammar FIRRTL;

tokens { INDENT, DEDENT }

@lexer::header {
import firrtl.LexerHelper;
}

@lexer::members {
  private final LexerHelper denter = new firrtl.LexerHelper()
  {
    @Override
    public Token pullToken() {
      return FIRRTLLexer.super.nextToken();
    }
  };

  @Override
  public Token nextToken() {
    return denter.nextToken();
  }
}

/*------------------------------------------------------------------
 * PARSER RULES
 *------------------------------------------------------------------*/

/* TODO 
 *  - Add [info] support (all over the place)
 *  - Add support for extmodule
*/

// Does there have to be at least one module?
circuit
  : 'circuit' id ':' info? INDENT module* DEDENT
  ;

module
  : 'module' id ':' info? INDENT port* moduleBlock DEDENT
  | 'extmodule' id ':' info? INDENT port* defname? parameter* DEDENT
  ;

port
  : dir id ':' type info? NEWLINE
  ;

dir
  : 'input'
  | 'output'
  ;

type 
  : 'UInt' ('<' intLit '>')?
  | 'SInt' ('<' intLit '>')?
  | 'Fixed' ('<' intLit '>')? ('<' '<' intLit '>' '>')?
  | 'Clock'
  | 'Analog' ('<' intLit '>')?
  | '{' field* '}'        // Bundle
  | type '[' intLit ']'   // Vector
  ;

field
  : 'flip'? fieldId ':' type
  ;

defname
  : 'defname' '=' id NEWLINE
  ;

parameter
  : 'parameter' id '=' intLit NEWLINE
  | 'parameter' id '=' StringLit NEWLINE
  | 'parameter' id '=' DoubleLit NEWLINE
  | 'parameter' id '=' RawString NEWLINE
  ;

moduleBlock
  : simple_stmt*
  ;

simple_reset0:  'reset' '=>' '(' exp exp ')';

simple_reset
	: simple_reset0
	| '(' simple_reset0 ')'
	;

reset_block
	: INDENT simple_reset info? NEWLINE DEDENT
	| '(' +  simple_reset + ')'
  ;

stmt
  : 'wire' id ':' type info?
  | 'reg' id ':' type exp ('with' ':' reset_block)? info?
  | 'mem' id ':' info? INDENT memField* DEDENT
  | 'cmem' id ':' type info?
  | 'smem' id ':' type info?
  | mdir 'mport' id '=' id '[' exp ']' exp info?
  | 'inst' id 'of' id info?
  | 'node' id '=' exp info?
  | exp '<=' exp info?
  | exp '<-' exp info?
  | exp 'is' 'invalid' info?
  | when
  | 'stop(' exp exp intLit ')' info?
  | 'printf(' exp exp StringLit ( exp)* ')' info?
  | 'skip' info?
  | 'attach' '(' exp+ ')' info?
  ;

memField
	:  'data-type' '=>' type NEWLINE
	| 'depth' '=>' intLit NEWLINE
	| 'read-latency' '=>' intLit NEWLINE
	| 'write-latency' '=>' intLit NEWLINE
	| 'read-under-write' '=>' ruw NEWLINE
	| 'reader' '=>' id+ NEWLINE
	| 'writer' '=>' id+ NEWLINE
	| 'readwriter' '=>' id+ NEWLINE
	;

simple_stmt
  : stmt | NEWLINE
  ;

/*
    We should provide syntatctical distinction between a "moduleBody" and a "suite":
    - statements require a "suite" which means they can EITHER have a "simple statement" (one-liner) on the same line
        OR a group of one or more _indented_ statements after a new-line. A "suite" may _not_ be empty
    - modules on the other hand require a group of one or more statements without any indentation to follow "port"
        definitions. Let's call that _the_ "moduleBody". A "moduleBody" could possibly be empty
*/
suite
  : simple_stmt
  | INDENT simple_stmt+ DEDENT
  ;

when
  : 'when' exp ':' info? suite? ('else' ( when | ':' info? suite?) )?
  ;

info
  : FileInfo
  ;

mdir
  : 'infer'
  | 'read'
  | 'write'
  | 'rdwr'
  ;

ruw
  : 'old'
  | 'new'
  | 'undefined'
  ;

exp
  : 'UInt' ('<' intLit '>')? '(' intLit ')'
  | 'SInt' ('<' intLit '>')? '(' intLit ')'
  | id    // Ref
  | exp '.' fieldId
  | exp '.' DoubleLit // TODO Workaround for #470
  | exp '[' intLit ']'
  | exp '[' exp ']'
  | 'mux(' exp exp exp ')'
  | 'validif(' exp exp ')'
  | primop exp* intLit*  ')'
  ;

id
  : Id
  | keywordAsId
  ;

fieldId
  : Id
  | RelaxedId
  | UnsignedInt
  | keywordAsId
  ;

intLit
  : UnsignedInt
  | SignedInt
  | HexLit
  ;

// Keywords that are also legal ids
keywordAsId
  : 'circuit'
  | 'module'
  | 'extmodule'
  | 'parameter'
  | 'input'
  | 'output'
  | 'UInt'
  | 'SInt'
  | 'Clock'
  | 'Analog'
  | 'Fixed'
  | 'flip'
  | 'wire'
  | 'reg'
  | 'with'
  | 'reset'
  | 'mem'
  | 'depth'
  | 'reader'
  | 'writer'
  | 'readwriter'
  | 'inst'
  | 'of'
  | 'node'
  | 'is'
  | 'invalid'
  | 'when'
  | 'else'
  | 'stop'
  | 'printf'
  | 'skip'
  | 'old'
  | 'new'
  | 'undefined'
  | 'mux'
  | 'validif'
  | 'cmem'
  | 'smem'
  | 'mport'
  | 'infer'
  | 'read'
  | 'write'
  | 'rdwr'
  ;

// Parentheses are added as part of name because semantics require no space between primop and open parentheses
// (And ANTLR either ignores whitespace or considers it everywhere)
primop
  : 'add('
  | 'sub('
  | 'mul('
  | 'div('
  | 'rem('
  | 'lt('
  | 'leq('
  | 'gt('
  | 'geq('
  | 'eq('
  | 'neq('
  | 'pad('
  | 'asUInt('
  | 'asSInt('
  | 'asClock('
  | 'shl('
  | 'shr('
  | 'dshl('
  | 'dshr('
  | 'cvt('
  | 'neg('
  | 'not('
  | 'and('
  | 'or('
  | 'xor('
  | 'andr('
  | 'orr('
  | 'xorr('
  | 'cat('
  | 'bits('
  | 'head('
  | 'tail('
  | 'asFixedPoint('
  | 'bpshl('
  | 'bpshr('
  | 'bpset('
  ;

/*------------------------------------------------------------------
 * LEXER RULES
 *------------------------------------------------------------------*/

UnsignedInt
  : '0'
  | PosInt
  ;

SignedInt
  : ( '+' | '-' ) PosInt
  ;

fragment
PosInt
  : [1-9] ( Digit )*
  ;

HexLit
  : '"' 'h' ( '+' | '-' )? ( HexDigit )+ '"'
  ;

DoubleLit
  : ( '+' | '-' )? Digit+ '.' Digit+ ( 'E' ( '+' | '-' )? Digit+ )?
  ;

fragment
Digit
  : [0-9]
  ;

fragment
HexDigit
  : [a-fA-F0-9]
  ;

StringLit
  : '"' UnquotedString? '"'
  ;

RawString
  : '\'' UnquotedString? '\''
  ;

fragment
UnquotedString
  : ( '\\\'' | '\\"' | ~[\r\n] )+?
  ;

FileInfo
  : '@[' ('\\]'|.)*? ']'
  ;

Id
  : LegalStartChar (LegalIdChar)*
  ;

RelaxedId
  : (LegalIdChar)+
  ;

fragment
LegalIdChar
  : LegalStartChar
  | Digit
  | '$'
  ;

fragment
LegalStartChar
  : [a-zA-Z_]
  ;

fragment COMMENT
  : ';' ~[\r\n]*
  ;

fragment WHITESPACE
	: [ \t,]+
	;

SKIP_
	: ( WHITESPACE | COMMENT ) -> skip
	;

NEWLINE
	:'\r'? '\n' ' '*
	;
