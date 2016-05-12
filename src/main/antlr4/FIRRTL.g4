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

/*------------------------------------------------------------------
 * PARSER RULES
 *------------------------------------------------------------------*/

/* TODO 
 *  - Add [info] support (all over the place)
 *  - Add support for extmodule
*/

// Does there have to be at least one module?
circuit
  : 'circuit' id ':' info? '{' module* '}'
  ;

module
  : 'module' id ':' info? '{' port* block '}'
  | 'extmodule' id ':' info? '{' port* '}'
  ;

port
  : dir id ':' type info?
  ;

dir
  : 'input'
  | 'output'
  ;

type 
  : 'UInt' ('<' IntLit '>')?
  | 'SInt' ('<' IntLit '>')?
  | 'Clock'
  | '{' field* '}'        // Bundle
  | type '[' IntLit ']'   // Vector
  ;

field
  : 'flip'? id ':' type
  ;

// Much faster than replacing block with stmt+
block
  : (stmt)*
  ; 

stmt
  : 'wire' id ':' type info?
  | 'reg' id ':' type exp ('with' ':' '{' 'reset' '=>' '(' exp exp ')' '}')? info?
  | 'mem' id ':' info? '{'
                     ( 'data-type' '=>' type
                     | 'depth' '=>' IntLit
                     | 'read-latency' '=>' IntLit
                     | 'write-latency' '=>' IntLit
                     | 'read-under-write' '=>' ruw
                     | 'reader' '=>' id
                     | 'writer' '=>' id
                     | 'readwriter' '=>' id
                     )*
                 '}'
  | 'cmem' id ':' type info?
  | 'smem' id ':' type info?
  | mdir 'mport' id '=' id '[' exp ']' exp info?
  | 'inst' id 'of' id info?
  | 'node' id '=' exp info?
  | exp '<=' exp info?
  | exp '<-' exp info?
  | exp 'is' 'invalid' info?
  | 'when' exp ':' info? '{' block '}' ( 'else' ':' '{' block '}' )?
  | 'stop(' exp exp IntLit ')' info?
  | 'printf(' exp exp StringLit (exp)* ')' info?
  | 'skip' info?
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

Comment
  : ';' ~[\r\n]* 
    -> skip
  ;

Whitespace
  : [ \t,]+
    -> skip
  ;

Newline 
  : ( '\r'? '\n' )+
      -> skip
  ;


