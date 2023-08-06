// RUN: circt-opt -firrtl-print-instance-graph %s -o %t 2>&1 | FileCheck %s

// CHECK: digraph "Top"
// CHECK:   label="Top";
// CHECK:   [[TOP:.*]] [shape=record,label="{Top}"];
// CHECK:   [[TOP]] -> [[ALLIGATOR:.*]][label=alligator];
// CHECK:   [[TOP]] -> [[CAT:.*]][label=cat];
// CHECK:   [[ALLIGATOR]] [shape=record,label="{Alligator}"];
// CHECK:   [[ALLIGATOR]] -> [[BEAR:.*]][label=bear];
// CHECK:   [[CAT]] [shape=record,label="{Cat}"];
// CHECK:   [[BEAR]] [shape=record,label="{Bear}"];
// CHECK:   [[BEAR]] -> [[CAT]][label=cat];

firrtl.circuit "Top" {

firrtl.module @Top() {
  firrtl.instance alligator @Alligator()
  firrtl.instance cat @Cat()
}

firrtl.module @Alligator() {
  firrtl.instance bear @Bear()
}

firrtl.module @Bear() {
  firrtl.instance cat @Cat()
}

firrtl.module @Cat() { }

}
