// RUN: circt-opt %s -split-input-file -verify-diagnostics

module attributes {calyx.entrypoint = "main"} {}

// -----

module attributes {calyx.entrypoint = "foo"} {
  calyx.component @bar(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @foo(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.instance' op cannot reference the entry-point component: 'main'.}}
    calyx.instance @c of @main : i16, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @foo(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    // expected-note @+1 {{see existing symbol definition here}}
    calyx.instance @c of @foo : i1, i1, i1, i1
    // expected-error @+1 {{redefinition of symbol named 'c'}}
    calyx.instance @c of @foo : i1, i1, i1, i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @foo(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    // expected-error @+1 {{instance symbol: 'foo' is already a symbol for another component.}}
    calyx.instance @foo of @foo : i1, i1, i1, i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  // expected-error @+1 {{'calyx.component' op requires exactly one of each: 'calyx.wires', 'calyx.control'.}}
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{referencing component: 'A', which does not exist.}}
    calyx.instance @c of @A
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{recursive instantiation of its parent component: 'main'}}
    calyx.instance @c of @main : i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.instance' op has a wrong number of results; expected: 5 but got 0}}
    calyx.instance @a0 of @A
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @B(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.instance' op result type for "in" must be 'i16', but got 'i1'}}
    %b0.in, %b0.go, %b0.clk, %b0.reset, %b0.done = calyx.instance @b of @B : i1, i1, i1, i1, i1
    calyx.wires { calyx.assign %done = %b0.done : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i16, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @B(%in: i16, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %a.go, %a.clk, %a.reset, %a.out, %a.done = calyx.instance @a of @A : i1, i1, i1, i16, i1
    %b.in, %b.go, %b.clk, %b.reset, %b.done = calyx.instance @b of @B : i16, i1, i1, i1, i1
    // expected-error @+1 {{'calyx.assign' op expects parent op to be one of 'calyx.group, calyx.comb_group, calyx.static_group, calyx.wires'}}
    calyx.assign %b.in = %a.out : i16

    calyx.wires { calyx.assign %b.in = %a.out : i16 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    calyx.wires {}
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.enable' op with group 'WrongName', which does not exist.}}
        calyx.enable @WrongName
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // expected-error @+1 {{'calyx.control' op EnableOp is not a composition operator. It should be nested in a control flow operation, such as "calyx.seq"}}
    calyx.control {
      calyx.enable @A
      calyx.enable @A
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with name: "Group1" is unused in the control execution schedule}}
      calyx.group @Group1 {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
      calyx.assign %done = %c1_1 : i1
    }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1
    calyx.wires {
      calyx.assign %done = %r.done : i1
    }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i1, i1, i1, i1, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Group1 {
        calyx.assign %c0.go = %c1_1 : i1
      }
      calyx.group @Group2 {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{empty 'else' region.}}
        calyx.if %c0.out with @Group1 {
          calyx.enable @Group2
        } else {}
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i1, i1, i1, i1, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    calyx.wires {
      calyx.group @Group2 {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.if' op with group 'Group1', which does not exist.}}
        calyx.if %c0.out with @Group1 { calyx.enable @Group2}
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%in: i1, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i1, i1, i1, i1, i1, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Group1 {}
      calyx.group @Group2 {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{conditional op: '%c0.out' expected to be driven from group: 'Group1' but no driver was found.}}
        calyx.if %c0.out with @Group1 {
          calyx.enable @Group2
        }
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i1, i1, i1, i1, i1
    calyx.wires { calyx.comb_group @Group2 { calyx.assign %c0.go = %c1_1 : i1 } }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.while' op with group 'Group1', which does not exist.}}
        calyx.while %c0.out with @Group1 {
          calyx.enable @Group2
        }
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%in: i1, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.in, %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i1, i1, i1, i1, i1, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.comb_group @Group1 { }
      calyx.group @Group2 {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{conditional op: '%c0.out' expected to be driven from group: 'Group1' but no driver was found.}}
        calyx.while %c0.out with @Group1 {
          calyx.enable @Group2
        }
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.memory' op mismatched number of dimensions (1) and address sizes (2)}}
    %m.addr0, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64] x 8> [6, 6] : i6, i8, i1, i1, i8, i1
    calyx.wires { calyx.assign %done = %m.done : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.memory' op incorrect number of address ports, expected 2}}
    %m.addr0, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64, 64] x 8> [6, 6] : i6, i8, i1, i1, i8, i1
    calyx.wires { calyx.assign %done = %m.done : i1}
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.memory' op address size (5) for dimension 0 can't address the entire range (64)}}
    %m.addr0, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64] x 8> [5] : i5, i8, i1, i1, i5, i1
    calyx.wires { calyx.assign %done = %m.done : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has an invalid destination port. It must be drive-able.}}
      calyx.assign %c1_1 = %go : i1
    }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has a component port as the destination with the incorrect direction.}}
      calyx.assign %go = %c1_1 : i1
    }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has a component port as the source with the incorrect direction.}}
      calyx.assign %r.write_en = %done : i1
    }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has a cell port as the source with the incorrect direction.}}
      calyx.assign %done = %r.write_en : i1
    }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op has a cell port as the destination with the incorrect direction.}}
      calyx.assign %r.done = %go : i1
    }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    // expected-error @+1 {{'calyx.control' op has an invalid control sequence. Multiple control flow operations must all be nested in a single calyx.seq or calyx.par}}
    calyx.control {
      calyx.seq { calyx.enable @A }
      calyx.seq { calyx.enable @A }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      // expected-error @+1 {{'calyx.par' op cannot enable the same group: "A" more than once.}}
      calyx.par {
        calyx.enable @A
        calyx.enable @A
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @Group1 {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.if' op with group 'Group1', which is not a combinational group.}}
        calyx.if %c0.out with @Group1 { calyx.enable @Group1}
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i1, i1, i1, i1, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @Group1 {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        // expected-error @+1 {{'calyx.while' op with group 'Group1', which is not a combinational group.}}
        calyx.while %c0.out with @Group1 { calyx.enable @Group1}
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @A(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out: i1, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0.go, %c0.clk, %c0.reset, %c0.out, %c0.done = calyx.instance @c0 of @A : i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.comb_group @Group1 { calyx.assign %c0.go = %c1_1 : i1 } }
    calyx.control {
      calyx.seq {
        calyx.if %c0.out {
          // expected-error @+1 {{'calyx.enable' op with group 'Group1', which is a combinational group.}}
          calyx.enable @Group1
        }
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  // expected-error @+1 {{'calyx.component' op is missing the following required port attribute identifiers: done, go}}
  calyx.component @main(%go: i1, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with cell: calyx.register "r" is performing a write and failed to drive all necessary ports.}}
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    %c1_i8 = hw.constant 1 : i8
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with cell: calyx.memory "m" is performing a write and failed to drive all necessary ports.}}
      calyx.group @A {
        calyx.assign %m.write_data = %c1_i8 : i8
        calyx.group_done %m.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %gt.left, %gt.right, %gt.out = calyx.std_gt @gt : i8, i8, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_i8 = hw.constant 1 : i8
    %c1_i1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with cell: calyx.std_gt "gt" is performing a write and failed to drive all necessary ports.}}
      calyx.group @A {
        calyx.assign %r.in = %gt.out : i1
        calyx.assign %r.write_en = %c1_i1 : i1
        calyx.assign %gt.left = %c1_i8 : i8
        calyx.group_done %r.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i8, i1, i1, i1, i8, i1
    %c1_i1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.group' op with cell: calyx.memory "m" is having a read performed upon it, and failed to drive all necessary ports.}}
      calyx.group @A {
        calyx.assign %r.in = %m.read_data : i8
        calyx.assign %r.write_en = %c1_i1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.comb_group' op with register: "r" is conducting a memory store. This is not combinational.}}
      calyx.comb_group @IncorrectCombGroup {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.if %r.out with @IncorrectCombGroup {
          calyx.enable @A
        }
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %m.addr0, %m.addr1, %m.write_data, %m.write_en, %m.clk, %m.read_data, %m.done = calyx.memory @m <[64, 64] x 8> [6, 6] : i6, i6, i8, i1, i1, i8, i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_i8 = hw.constant 1 : i8
    %c0_i6 = hw.constant 0 : i6
    %c1_i1 = hw.constant 1 : i1
    calyx.wires {
      // expected-error @+1 {{'calyx.comb_group' op with memory: "m" is conducting a memory store. This is not combinational.}}
      calyx.comb_group @IncorrectCombGroup {
        calyx.assign %m.write_data = %c1_i8 : i8
        calyx.assign %m.addr0 = %c0_i6 : i6
        calyx.assign %m.addr1 = %c0_i6 : i6
        calyx.assign %m.write_en = %c1_i1 : i1
      }
      calyx.group @A {
        calyx.assign %r.in = %c1_i1 : i1
        calyx.assign %r.write_en = %c1_i1 : i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.if %r.out with @IncorrectCombGroup {
          calyx.enable @A
        }
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        %and = comb.and %c1_1, %c1_1 : i1
        // expected-error @+1 {{'calyx.assign' op has source that is not a port or constant. Complex logic should be conducted in the guard.}}
        calyx.assign %r.in = %c1_1 ? %and : i1
        calyx.assign %r.write_en = %c1_1: i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1: i1
        %and = comb.and %c1_1, %c1_1 : i1
        // expected-error @+1 {{'calyx.group_done' op has source that is not a port or constant. Complex logic should be conducted in the guard.}}
        calyx.group_done %and : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        %and = comb.and %c1_1, %c1_1 : i1
        // expected-error @+1 {{'calyx.group_go' op has source that is not a port or constant. Complex logic should be conducted in the guard.}}
        calyx.group_go %and : i1
        calyx.assign %r.in = %c1_1 : i1
        calyx.assign %r.write_en = %c1_1: i1
        calyx.group_done %r.done : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    calyx.wires {
      calyx.group @A {
        // expected-error @+1 {{'calyx.group_done' op with constant source and constant guard. This should be a combinational group.}}
        calyx.group_done %c1_1 ? %c1_1 : i1
      }
    }
    calyx.control { calyx.enable @A }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.std_slice' op expected input bits (32) to be greater than output bits (64)}}
    %std_slice.in, %std_slice.out = calyx.std_slice @std_slice : i32, i64
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    // expected-error @+1 {{'calyx.std_pad' op expected input bits (64) to be less than output bits (32)}}
    %std_pad.in, %std_pad.out = calyx.std_pad @std_pad : i64, i32
    %c1_1 = hw.constant 1 : i1
    calyx.wires { calyx.assign %done = %c1_1 : i1 }
    calyx.control {}
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    calyx.wires {
      calyx.assign %std_lt_0.left = %c64_i32 : i32
      // expected-error @+1 {{'calyx.assign' op destination is already continuously driven. Other assignment is "calyx.assign"(%0#0, %1) : (i32, i32) -> ()}}
      calyx.assign %std_lt_0.left = %c42_i32 : i32
    }
    calyx.control {
      calyx.seq { }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%cond: i1, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    calyx.wires {
      // expected-error @+1 {{'calyx.assign' op destination is already continuously driven. Other assignment is "calyx.assign"(%0#0, %2) : (i32, i32) -> ()}}
      calyx.assign %std_lt_0.left = %cond ? %c64_i32 : i32
      calyx.assign %std_lt_0.left = %c42_i32 : i32
    }
    calyx.control {
      calyx.seq { }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    calyx.wires {
      calyx.assign %std_lt_0.left = %c64_i32 : i32
      calyx.group @A {
      // expected-error @+1 {{'calyx.assign' op destination is already continuously driven. Other assignment is "calyx.assign"(%0#0, %1) : (i32, i32) -> ()}}
        calyx.assign %std_lt_0.left = %c42_i32 : i32
        calyx.assign %std_lt_0.right = %c42_i32 : i32
        calyx.group_done %r.done : i1
      }
    }
    calyx.control {
      calyx.seq {
        calyx.enable @A
      }
    }
  }
}


// -----

module attributes {calyx.entrypoint = "main"} {
  // expected-error @+1 {{'calyx.component' op The component currently does nothing. It needs to either have continuous assignments in the Wires region or control constructs in the Control region.}}
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i1, i1, i1, i1, i1, i1
    calyx.wires {
    }
    calyx.control {
    }
  }
}

// -----
module attributes {calyx.entrypoint = "A"} {
  hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>, %clk: i1 {calyx.clk}, %go: i1 {calyx.go = 1}) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>, done: i1 {calyx.done}) attributes {filename = "test.v"}

  calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    // expected-error @+1 {{'calyx.primitive' op has the wrong number of parameters; expected: 1 but got 0}}
    %params.in, %params.clk, %params.go, %params.out, %params.done = calyx.primitive @params_0 of @params : i32, i1, i1, i32, i1

    calyx.wires {
      calyx.assign %done = %c1_1 : i1
      calyx.assign %params.in = %in_0 : i32
      calyx.assign %out_0 = %params.out : i32
    }
    calyx.control {}
  } {static = 1}
}

// -----
module attributes {calyx.entrypoint = "A"} {
  hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"WIDTH">>, %clk: i1 {calyx.clk}, %go: i1 {calyx.go = 1}) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>, done: i1 {calyx.done}) attributes {filename = "test.v"}

  calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    // expected-error @+1 {{'calyx.primitive' op parameter #0 should have name "WIDTH" but has name "TEST"}}
    %params.in, %params.clk, %params.go, %params.out, %params.done = calyx.primitive @params_0 of @params<TEST: i32 = 1> : i32, i1, i1, i32, i1

    calyx.wires {
      calyx.assign %done = %c1_1 : i1
      calyx.assign %params.in = %in_0 : i32
      calyx.assign %out_0 = %params.out : i32
    }
    calyx.control {}
  } {static = 1}
}

// -----
module attributes {calyx.entrypoint = "A"} {
  hw.module.extern @params<WIDTH: i32>(%in: !hw.int<#hw.param.decl.ref<"TEST">>, %clk: i1 {calyx.clk}, %go: i1 {calyx.go = 1}) -> (out: !hw.int<#hw.param.decl.ref<"WIDTH">>, done: i1 {calyx.done}) attributes {filename = "test.v"}

  calyx.component @A(%in_0: i32, %in_1: i32, %go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%out_0: i32, %out_1: i32, %done: i1 {done}) {
    %c1_1 = hw.constant 1 : i1
    // expected-error @+1 {{Could not find parameter TEST in the provided parameters for the expression!}}
    %params.in, %params.clk, %params.go, %params.out, %params.done = calyx.primitive @params_0 of @params<WIDTH: i32 = 1> : i32, i1, i1, i32, i1

    calyx.wires {
      calyx.assign %done = %c1_1 : i1
      calyx.assign %params.in = %in_0 : i32
      calyx.assign %out_0 = %params.out : i32
    }
    calyx.control {}
  } {static = 1}
}

// -----

module attributes {calyx.entrypoint = "main"} {
  // expected-error @+1 {{'calyx.comb_component' op must not have a `calyx.control` op.}}
  calyx.comb_component @main() -> () {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    calyx.wires {
      calyx.assign %std_lt_0.left = %c64_i32 : i32
    }
    calyx.control {
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  // expected-error @+1 {{'calyx.comb_component' op requires exactly one calyx.wires op.}}
  calyx.comb_component @main() -> () {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    calyx.wires {
      calyx.assign %std_lt_0.left = %c64_i32 : i32
    }
    calyx.wires {
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  // expected-error @+1 {{'calyx.comb_component' op The component currently does nothing. It needs to either have continuous assignments in the Wires region or control constructs in the Control region.}}
  calyx.comb_component @main() -> () {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    calyx.wires {
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  // expected-error @+1 {{'calyx.comb_component' op contains non-combinational cell mu}}
  calyx.comb_component @main() -> () {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    %mu.clk, %mu.reset, %mu.go, %mu.left, %mu.right, %mu.out, %mu.done = calyx.std_mult_pipe @mu : i1, i1, i1, i32, i32, i32, i1
    calyx.wires {
      calyx.assign %std_lt_0.left = %c64_i32 : i32
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  // expected-error @+1 {{'calyx.comb_component' op contains group A}}
  calyx.comb_component @main() -> (%out: i32) {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    %true = hw.constant 1 : i1
    calyx.wires {
      calyx.assign %out = %c42_i32 : i32
      calyx.group @A {
        calyx.assign %std_lt_0.left = %c42_i32 : i32
        calyx.assign %std_lt_0.right = %c42_i32 : i32
        calyx.group_done %true : i1
      }
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  // expected-error @+1 {{'calyx.comb_component' op contains comb group A}}
  calyx.comb_component @main() -> (%out: i32) {
    %std_lt_0.left, %std_lt_0.right, %std_lt_0.out = calyx.std_lt @std_lt_0 : i32, i32, i1
    %c64_i32 = hw.constant 64 : i32
    %c42_i32 = hw.constant 42 : i32
    calyx.wires {
      calyx.assign %out = %c42_i32 : i32
      calyx.comb_group @A {
        calyx.assign %std_lt_0.left = %c42_i32 : i32
        calyx.assign %std_lt_0.right = %c42_i32 : i32
      }
    }
  }
}

// ----- 

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
    calyx.wires {

    }
    calyx.control {
      // expected-error @+1 {{'calyx.invoke' op '@r' has zero input and output port connections; expected at least one.}}
      calyx.invoke @r() -> ()
    }
  }
}

// ----- 

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c10 = hw.constant 10 : i32
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1 
    calyx.wires {

    }
    calyx.control {
      // expected-error @+2 {{'calyx.invoke' op has a cell port as the destination with the incorrect direction.}} 
      // expected-error @+1 {{'calyx.invoke' op '@r' has input '%r.out', which is a source port. The inputs are required to be destination ports.}}
      calyx.invoke @r(%r.out = %c10) -> (i32)
      
    }
  }
}

// ----- 

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c10 = hw.constant 10 : i32
    %r0.in, %r0.write_en, %r0.clk, %r0.reset, %r0.out, %r0.done = calyx.register @r0 : i32, i1, i1, i1, i32, i1 
    %r1.in, %r1.write_en, %r1.clk, %r1.reset, %r1.out, %r1.done = calyx.register @r1 : i32, i1, i1, i1, i32, i1
    calyx.wires {

    }
    calyx.control {
      // expected-error @+2 {{'calyx.invoke' op has a cell port as the source with the incorrect direction.}} 
      // expected-error @+1 {{'calyx.invoke' op '@r0' has output '%r1.in', which is a destination port. The inputs are required to be source ports.}}
      calyx.invoke @r0(%r0.in = %r1.in) -> (i32)
    }
  }
}

// ----- 

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1 = hw.constant 1 : i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1 
    calyx.wires {

    }
    calyx.control {
      // expected-error @+1 {{'calyx.invoke' op the go or write_en port of '@r' cannot appear here.}}
      calyx.invoke @r(%r.write_en = %c1) -> (i1)
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c1 = hw.constant 1 : i1
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1 
    calyx.wires {

    }
    calyx.control {
      // expected-error @+1 {{'calyx.invoke' op the done port of '@r' cannot appear here.}}
      calyx.invoke @r(%done = %r.done) -> (i1)
    }
  }
}

// ----- 

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c10 = hw.constant 10 : i32
    %add.left, %add.right, %add.out = calyx.std_add @add : i32, i32, i32
    calyx.wires {

    }
    calyx.control {
      // expected-error @+1 {{'calyx.invoke' op '@add' is a combinational component and cannot be invoked, which must have single go port and single done port.}}
      calyx.invoke @add(%add.left = %c10, %add.right = %c10) -> (i32, i32)
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}, %in : i32) -> (%done: i1 {done}) {
    %c10 = hw.constant 10 : i32
    %r0.in, %r0.write_en, %r0.clk, %r0.reset, %r0.out, %r0.done = calyx.register @r0 : i32, i1, i1, i1, i32, i1 
    %r1.in, %r1.write_en, %r1.clk, %r1.reset, %r1.out, %r1.done = calyx.register @r1 : i32, i1, i1, i1, i32, i1
    calyx.wires {
      
    }
    calyx.control {
      // expected-error @+1 {{'calyx.invoke' op the connection %r1.in = %c10 is not defined as an input port of '@r0'.}}
      calyx.invoke @r0(%r1.in = %c10) -> (i32)
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    calyx.wires {

    }
    calyx.control {
      // expected-error @+1 {{'calyx.invoke' op with instance '@comp', which does not exist.}}
      calyx.invoke @comp() -> ()
    }
  }
}

// -----

module attributes {calyx.entrypoint = "main"} {
  calyx.component @main(%go: i1 {go}, %clk: i1 {clk}, %reset: i1 {reset}) -> (%done: i1 {done}) {
    %c0 = hw.constant 0 : i32 
    %c1 = hw.constant 1 : i32  
    %and = comb.and %c0, %c1 : i32 
    %r.in, %r.write_en, %r.clk, %r.reset, %r.out, %r.done = calyx.register @r : i32, i1, i1, i1, i32, i1
    calyx.wires {

    }
    calyx.control {
      // expected-error @+1 {{'calyx.invoke' op '@r' has '%and', which is not a port or constant. Complex logic should be conducted in the guard.}}
      calyx.invoke @r(%r.in = %and) -> (i32)
    }
  }
}
