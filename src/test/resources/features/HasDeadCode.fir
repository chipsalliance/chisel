; SPDX-License-Identifier: Apache-2.0
circuit HasDeadCode :
  module HasDeadCode :
    input i : UInt<1>
    output o : UInt<1>
    wire a : UInt<1>
    a <= not(i)
    o <= i
