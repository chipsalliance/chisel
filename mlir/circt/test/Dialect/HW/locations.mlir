// RUN: circt-opt %s | circt-opt | FileCheck %s -check-prefix BASIC
// RUN: circt-opt %s --mlir-print-debuginfo | circt-opt --mlir-print-debuginfo --mlir-print-local-scope | FileCheck %s -check-prefix DEBUG

// Basic IR parser+printer round tripping test. The debug locations should only
// be printed when the required, and they should be accurately parsed back in.

hw.module @test0(%input: i7) -> (output: i7) { hw.output %input : i7 }
hw.module @test1(%input: i7 {hw.arg = "arg"}) -> (output: i7 {hw.res = "res"}) { hw.output %input : i7 }
hw.module @test2(%input: i7 loc("arg")) -> (output: i7 loc("res")) { hw.output %input : i7 }
hw.module @test3(%input: i7 {hw.arg = "arg"} loc("arg")) -> (output: i7 {hw.res = "res"} loc("res")) { hw.output %input : i7 }
hw.module.extern @test4(%input: i7) -> (output: i7)
hw.module.extern @test5(%input: i7 {hw.arg = "arg"}) -> (output: i7 {hw.res = "res"})
hw.module.extern @test6(%input: i7 loc("arg")) -> (output: i7 loc("res"))
hw.module.extern @test7(%input: i7 {hw.arg = "arg"} loc("arg")) -> (output: i7 {hw.res = "res"} loc("res"))

// BASIC: hw.module @test0(%input: i7) -> (output: i7)
// BASIC: hw.module @test1(%input: i7 {hw.arg = "arg"}) -> (output: i7 {hw.res = "res"})
// BASIC: hw.module @test2(%input: i7) -> (output: i7)
// BASIC: hw.module @test3(%input: i7 {hw.arg = "arg"}) -> (output: i7 {hw.res = "res"})
// BASIC: hw.module.extern @test4(%input: i7) -> (output: i7)
// BASIC: hw.module.extern @test5(%input: i7 {hw.arg = "arg"}) -> (output: i7 {hw.res = "res"})
// BASIC: hw.module.extern @test6(%input: i7) -> (output: i7)
// BASIC: hw.module.extern @test7(%input: i7 {hw.arg = "arg"}) -> (output: i7 {hw.res = "res"})

// DEBUG: hw.module @test0(%input: i7 loc({{.+}})) -> (output: i7 loc({{.+}}))
// DEBUG: hw.module @test1(%input: i7 {hw.arg = "arg"} loc({{.+}})) -> (output: i7 {hw.res = "res"} loc({{.+}}))
// DEBUG: hw.module @test2(%input: i7 loc("arg")) -> (output: i7 loc("res"))
// DEBUG: hw.module @test3(%input: i7 {hw.arg = "arg"} loc("arg")) -> (output: i7 {hw.res = "res"} loc("res"))
// DEBUG: hw.module.extern @test4(%input: i7 loc({{.+}})) -> (output: i7 loc({{.+}}))
// DEBUG: hw.module.extern @test5(%input: i7 {hw.arg = "arg"} loc({{.+}})) -> (output: i7 {hw.res = "res"} loc({{.+}}))
// DEBUG: hw.module.extern @test6(%input: i7 loc("arg")) -> (output: i7 loc("res"))
// DEBUG: hw.module.extern @test7(%input: i7 {hw.arg = "arg"} loc("arg")) -> (output: i7 {hw.res = "res"} loc("res"))
