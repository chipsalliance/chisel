// RUN: circt-opt %s --split-input-file --verify-diagnostics

// expected-error @+1 {{unknown  type `illegaltype` in dialect `llhd`}}
func.func @illegaltype(%arg0: !llhd.illegaltype) {
    return
}

// -----

// expected-error @+2 {{unknown attribute `illegalattr` in dialect `llhd`}}
func.func @illegalattr() {
    %0 = llhd.constant_time #llhd.illegalattr : i1
    return
}
