; SPDX-License-Identifier: Apache-2.0
circuit AddNot:
  module AddNot:
    input a: UInt<8>
    input b: UInt<8>
    output o: UInt<9>
    o <= add(a, not(b))
