/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/
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
  | 'extmodule' id ':' info? INDENT port* DEDENT
  ;

port
  : dir id ':' type info? NEWLINE
  ;

dir
  : 'input'
  | 'output'
  ;

type 
  : 'UInt' ('<' IntLit '>')?
  | 'SInt' ('<' IntLit '>')?
  | 'Clock'
  | 'Analog' ('<' IntLit '>')?
  | '{' field* '}'        // Bundle
  | type '[' IntLit ']'   // Vector
  ;

field
  : 'flip'? id ':' type
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
	: INDENT simple_reset NEWLINE DEDENT
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
  | 'stop(' exp exp IntLit ')' info?
  | 'printf(' exp exp StringLit ( exp)* ')' info?
  | 'skip' info?
  | 'attach' exp 'to' '(' exp* ')' info?
  ;

memField
	:  'data-type' '=>' type NEWLINE
	| 'depth' '=>' IntLit NEWLINE
	| 'read-latency' '=>' IntLit NEWLINE
	| 'write-latency' '=>' IntLit NEWLINE
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
  : 'UInt' ('<' IntLit '>')? '(' IntLit ')' 
  | 'SInt' ('<' IntLit '>')? '(' IntLit ')' 
  | 'UBits' ('<' IntLit '>')? '(' StringLit ')'
  | 'SBits' ('<' IntLit '>')? '(' StringLit ')'
  | id    // Ref
  | exp '.' id 
  | exp '[' IntLit ']'
  | exp '[' exp ']'
  | 'mux(' exp exp exp ')'
  | 'validif(' exp exp ')'
  | primop exp* IntLit*  ')' 
  ;

id
  : Id
  | keyword
  ;

keyword
  : 'circuit'
  | 'module'
  | 'extmodule'
  | 'input'
  | 'output'
  | 'UInt'
  | 'SInt'
  | 'UBits'
  | 'SBits'
  | 'Clock'
  | 'flip'
  | 'wire'
  | 'reg'
  | 'with'
  | 'reset'
  | 'mem'
  | 'data-type'
  | 'depth'
  | 'read-latency'
  | 'write-latency'
  | 'read-under-write'
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
  ;

/*------------------------------------------------------------------
 * LEXER RULES
 *------------------------------------------------------------------*/

IntLit
  : '0'
  | ( '+' | '-' )? [1-9] ( Digit )*
  | '"' 'h' ( HexDigit )+ '"'
  ;

fragment
Nondigit
  : [a-zA-Z_]
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
  : '"' ('\\"'|.)*? '"'
  ;

FileInfo
  : '@[' ('\\]'|.)*? ']'
  ;

Id
  : IdNondigit
    ( IdNondigit
    | Digit
    )*
  ;

fragment
IdNondigit
  : Nondigit
  | [~!@#$%^*\-+=?/]
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
