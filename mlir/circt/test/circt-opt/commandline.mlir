// RUN: circt-opt --help | FileCheck %s --check-prefix=HELP
// RUN: circt-opt --show-dialects | FileCheck %s --check-prefix=DIALECT

// HELP: OVERVIEW: CIRCT modular optimizer driver

// DIALECT: Available Dialects:
// DIALECT-SAME: affine
// DIALECT-SAME: arc
// DIALECT-SAME: arith
// DIALECT-SAME: builtin
// DIALECT-SAME: calyx
// DIALECT-SAME: cf
// DIALECT-SAME: chirrtl
// DIALECT-SAME: comb
// DIALECT-SAME: emitc
// DIALECT-SAME: esi
// DIALECT-SAME: firrtl
// DIALECT-SAME: fsm
// DIALECT-SAME: func
// DIALECT-SAME: handshake
// DIALECT-SAME: hw
// DIALECT-SAME: hwarith
// DIALECT-SAME: ibis
// DIALECT-SAME: interop
// DIALECT-SAME: llhd
// DIALECT-SAME: llvm
// DIALECT-SAME: memref
// DIALECT-SAME: moore
// DIALECT-SAME: msft
// DIALECT-SAME: om
// DIALECT-SAME: pipeline
// DIALECT-SAME: scf
// DIALECT-SAME: seq
// DIALECT-SAME: ssp
// DIALECT-SAME: sv
// DIALECT-SAME: systemc
