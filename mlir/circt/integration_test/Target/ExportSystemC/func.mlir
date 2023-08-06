// REQUIRES: clang-tidy
// RUN: circt-translate %s --export-systemc > %t.cpp
// RUN: clang-tidy --extra-arg=-frtti %t.cpp --

emitc.include <"stdint.h">
emitc.include <"functional">
emitc.include <"vector">

systemc.cpp.func private @funcDeclaration (%a: i32, %b: i32) -> i32

systemc.cpp.func externC private @declarationWithoutArgNames(i32, i8)

systemc.cpp.func @voidFunc () {
  systemc.cpp.return
}

systemc.cpp.func @testFunc (%a: i64, %b: i32) -> i32 {
  %0 = systemc.cpp.call @funcDeclaration(%b, %b) : (i32, i32) -> i32
  systemc.cpp.call @voidFunc() : () -> ()
  %v1 = systemc.cpp.variable : !emitc.opaque<"std::vector<int>">
  %pb = systemc.cpp.member_access %v1 dot "push_back" : (!emitc.opaque<"std::vector<int>">) -> ((!emitc.opaque<"int">) -> ())
  %c0 = "emitc.constant" () {value=#emitc.opaque<"0"> : !emitc.opaque<"int">} : () -> !emitc.opaque<"int">
  systemc.cpp.call_indirect %pb (%c0) : (!emitc.opaque<"int">) -> ()
  %2 = systemc.cpp.call @funcDeclaration(%b, %b) : (i32, i32) -> i32
  %v = systemc.cpp.variable %2 : i32
  systemc.cpp.return %0 : i32
}
