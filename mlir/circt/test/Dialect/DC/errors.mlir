// RUN: circt-opt %s -split-input-file -verify-diagnostics

hw.module @buffer(%0 : !dc.value<i1>) {
  // expected-error @+1 {{'dc.buffer' op expected 2 init values but got 1.}}
  %bufferInit = dc.buffer [2] %0 [1] : !dc.value<i1>
}

// -----

hw.module @types(%0 : i1) {
  // expected-error @+1 {{custom op 'dc.buffer' 'input' must be must be a !dc.value or !dc.token type, but got 'i1'}}
  %bufferInit = dc.buffer [2] %0 : i1
}
