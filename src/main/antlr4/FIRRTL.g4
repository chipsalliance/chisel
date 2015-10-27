grammar FIRRTL;

/*------------------------------------------------------------------
 * PARSER RULES
 *------------------------------------------------------------------*/

// TODO add [info] support (all over the place)
// TODO Fix connect
// TODO Add partial connect
// TODO Should FIRRTL keywords be legal IDs?

// Does there have to be at least one module?
circuit
  : 'circuit' id ':' '{' module* '}'
  ;

// TODO Add support for extmodule
module
  : 'module' id ':' '{' port* blockStmt '}'
  ;

port
  : portKind id ':' type
  ;

portKind
  : 'input'
  | 'output'
  ;

type 
  : 'UInt' ('<' width '>')?
  | 'SInt' ('<' width '>')?
  | 'Clock'
  | '{' field* '}'        // Bundle
  | type '[' IntLit ']'   // Vector
  ;

field
  : orientation id ':' type
  ;

// FIXME This is what the spec says it should be
//orientation
//  : 'default'
//  | 'reverse'
//  ;
orientation
  :  'flip'
  |  // Nothing
  ;

width
  : IntLit
  | '?'
  ;

// Much faster than replacing blockStmt with stmt+
blockStmt
  : (stmt)*
  ; 

stmt
  : 'wire' id ':' type
  | 'reg' id ':' type exp exp
  | 'smem' id ':' type exp
  | 'cmem' id ':' type exp
  | 'inst' id (':' | 'of') id // FIXME which should it be? ':' or 'of'
  | 'node' id '=' exp
  | 'poison' id ':' type        // Poison, FIXME
  | dir 'accessor' id '=' exp '[' exp ']' exp? // FIXME what is this extra exp?
  | exp ':=' exp                // Connect
  | 'onreset' exp ':=' exp      
  | exp '<>' exp                // Bulk Connect
  | exp '[' IntLit 'through' IntLit ']' ':=' exp   // SubWordConnect
  | 'when' exp ':' '{' blockStmt '}' ( 'else' ':' '{' blockStmt '}' )? 
  | 'assert' exp
  | 'skip'
  ;

// Accessor Direction
dir
  : 'infer'
  | 'read'
  | 'write'
  | 'rdwr'
  ;

// TODO implement
// What is exp?
exp
  : 'UInt' ('<' width '>')? '(' (IntLit) ')' // FIXME what does "ints" mean?
  | 'SInt' ('<' width '>')? '(' (IntLit) ')' // FIXME same
  | id    // Ref
  | exp '.' id // FIXME Does this work for no space?
  | exp '[' IntLit ']'
  | primop '(' exp* IntLit*  ')' // FIXME Need a big check here
  ;

id
  : Id
  | keyword
  ;

// FIXME need to make sure this is exhaustive including all FIRRTL keywords that are legal IDs
keyword
  : primop
  | dir
  | 'inst'  
  ;

primop
  : 'add'
  | 'sub'
  | 'addw'
  | 'subw'
  | 'mul'
  | 'div'
  | 'mod'
  | 'quo'
  | 'rem'
  | 'lt'
  | 'leq'
  | 'gt'
  | 'geq'
  | 'eq'
  | 'neq'
  | 'eqv'
  | 'neqv'
  | 'mux'
  | 'pad'
  | 'asUInt'
  | 'asSInt'
  | 'shl'
  | 'shr'
  | 'dshl'
  | 'dshr'
  | 'cvt'
  | 'neg'
  | 'not'
  | 'and'
  | 'or'
  | 'xor'
  | 'andr'
  | 'orr'
  | 'xorr'
  | 'cat'
  | 'bit'
  | 'bits'
  ;

/*------------------------------------------------------------------
 * LEXER RULES
 *------------------------------------------------------------------*/

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

// Should enforcing signed, non-neg, and positive ints be done in parser?
//   =>  YES
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
  : [a-zA-Z0-9]
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
