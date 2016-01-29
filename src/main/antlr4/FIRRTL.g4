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
  : 'circuit' id ':' '{' module* '}'
  ;

module
  : 'module' id ':' '{' port* block '}'
  ;

port
  : dir id ':' type
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
  : 'wire' id ':' type
  | 'reg' id ':' type exp (exp exp)?
  | 'mem' id ':' '{' ( 'data-type' '=>' type 
                     | 'depth' '=>' IntLit
                     | 'read-latency' '=>' IntLit
                     | 'write-latency' '=>' IntLit
                     | 'read-under-write' '=>' ruw
                     | 'reader' '=>' id
                     | 'writer' '=>' id
                     | 'readwriter' '=>' id 
                     )*
                 '}'
  | 'inst' id 'of' id 
  | 'node' id '=' exp
  | exp '<=' exp 
  | exp '<-' exp
  | exp 'is' 'invalid'
  | 'when' exp ':' '{' block '}' ( 'else' ':' '{' block '}' )? 
  | 'stop(' exp exp IntLit ')'
  | 'printf(' exp exp StringLit (exp)* ')'
  | 'skip'
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

// TODO add all keywords
keyword
  : dir
  | 'inst'  
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
  : '"' .*? '"'
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
  | [~!@#$%^*-+=?/]
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


