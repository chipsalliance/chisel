// RUN: circt-opt %s | circt-opt | FileCheck %s

%true = hw.constant true

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// CHECK: unrealized_conversion_cast to !ltl.sequence
// CHECK: unrealized_conversion_cast to !ltl.property
%s = unrealized_conversion_cast to !ltl.sequence
%p = unrealized_conversion_cast to !ltl.property

//===----------------------------------------------------------------------===//
// Generic
//===----------------------------------------------------------------------===//

// CHECK-NEXT: ltl.and {{%.+}}, {{%.+}} : i1, i1
// CHECK-NEXT: ltl.and {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence
// CHECK-NEXT: ltl.and {{%.+}}, {{%.+}} : !ltl.property, !ltl.property
ltl.and %true, %true : i1, i1
ltl.and %s, %s : !ltl.sequence, !ltl.sequence
ltl.and %p, %p : !ltl.property, !ltl.property

// CHECK-NEXT: ltl.or {{%.+}}, {{%.+}} : i1, i1
// CHECK-NEXT: ltl.or {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence
// CHECK-NEXT: ltl.or {{%.+}}, {{%.+}} : !ltl.property, !ltl.property
ltl.or %true, %true : i1, i1
ltl.or %s, %s : !ltl.sequence, !ltl.sequence
ltl.or %p, %p : !ltl.property, !ltl.property

// Type inference. `unrealized_conversion_cast` used to detect unexpected return
// types on `ltl.and`.
%s0 = ltl.and %true, %true : i1, i1
%s1 = ltl.and %true, %s : i1, !ltl.sequence
%s2 = ltl.and %s, %true : !ltl.sequence, i1
%p0 = ltl.and %true, %p : i1, !ltl.property
%p1 = ltl.and %p, %true : !ltl.property, i1
%p2 = ltl.and %s, %p : !ltl.sequence, !ltl.property
%p3 = ltl.and %p, %s : !ltl.property, !ltl.sequence
unrealized_conversion_cast %s0 : !ltl.sequence to index
unrealized_conversion_cast %s1 : !ltl.sequence to index
unrealized_conversion_cast %s2 : !ltl.sequence to index
unrealized_conversion_cast %p0 : !ltl.property to index
unrealized_conversion_cast %p1 : !ltl.property to index
unrealized_conversion_cast %p2 : !ltl.property to index
unrealized_conversion_cast %p3 : !ltl.property to index

//===----------------------------------------------------------------------===//
// Sequences
//===----------------------------------------------------------------------===//

// CHECK: ltl.delay {{%.+}}, 0 : !ltl.sequence
// CHECK: ltl.delay {{%.+}}, 42, 1337 : !ltl.sequence
ltl.delay %s, 0 : !ltl.sequence
ltl.delay %s, 42, 1337 : !ltl.sequence

// CHECK: ltl.concat {{%.+}} : !ltl.sequence
// CHECK: ltl.concat {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence
// CHECK: ltl.concat {{%.+}}, {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.sequence, !ltl.sequence
ltl.concat %s : !ltl.sequence
ltl.concat %s, %s : !ltl.sequence, !ltl.sequence
ltl.concat %s, %s, %s : !ltl.sequence, !ltl.sequence, !ltl.sequence

//===----------------------------------------------------------------------===//
// Properties
//===----------------------------------------------------------------------===//

// CHECK: ltl.not {{%.+}} : i1
// CHECK: ltl.not {{%.+}} : !ltl.sequence
// CHECK: ltl.not {{%.+}} : !ltl.property
ltl.not %true : i1
ltl.not %s : !ltl.sequence
ltl.not %p : !ltl.property

// CHECK: ltl.implication {{%.+}}, {{%.+}} : !ltl.sequence, !ltl.property
ltl.implication %s, %p : !ltl.sequence, !ltl.property

// CHECK: ltl.eventually {{%.+}} : i1
// CHECK: ltl.eventually {{%.+}} : !ltl.sequence
// CHECK: ltl.eventually {{%.+}} : !ltl.property
ltl.eventually %true : i1
ltl.eventually %s : !ltl.sequence
ltl.eventually %p : !ltl.property

//===----------------------------------------------------------------------===//
// Clocking
//===----------------------------------------------------------------------===//

// CHECK: ltl.clock {{%.+}}, posedge {{%.+}} : !ltl.sequence
// CHECK: ltl.clock {{%.+}}, negedge {{%.+}} : !ltl.sequence
// CHECK: ltl.clock {{%.+}}, edge {{%.+}} : i1
// CHECK: ltl.clock {{%.+}}, edge {{%.+}} : !ltl.sequence
// CHECK: ltl.clock {{%.+}}, edge {{%.+}} : !ltl.property
ltl.clock %s, posedge %true : !ltl.sequence
ltl.clock %s, negedge %true : !ltl.sequence
%clk0 = ltl.clock %true, edge %true : i1
%clk1 = ltl.clock %s, edge %true : !ltl.sequence
%clk2 = ltl.clock %p, edge %true : !ltl.property

// Type inference. `unrealized_conversion_cast` used to detect unexpected return
// types on `ltl.and`.
unrealized_conversion_cast %clk0 : !ltl.sequence to index
unrealized_conversion_cast %clk1 : !ltl.sequence to index
unrealized_conversion_cast %clk2 : !ltl.property to index

// CHECK: ltl.disable {{%.+}} if {{%.+}} : !ltl.property
ltl.disable %p if %true : !ltl.property
