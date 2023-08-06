// RUN: circt-opt -hw-print-instance-graph %s -o %t 2>&1 | FileCheck %s

// CHECK:   [[TOP:.*]] [shape=record,label="{Top}"];
// CHECK:   [[TOP]] -> [[ALLIGATOR:.*]][label=alligator];
// CHECK:   [[TOP]] -> [[CAT:.*]][label=cat];
// CHECK:   [[ALLIGATOR]] [shape=record,label="{Alligator}"];
// CHECK:   [[ALLIGATOR]] -> [[BEAR:.*]][label=bear];
// CHECK:   [[CAT]] [shape=record,label="{Cat}"];
// CHECK:   [[BEAR]] [shape=record,label="{Bear}"];
// CHECK:   [[BEAR]] -> [[CAT]][label=cat];

hw.module @Top() {
  hw.instance "alligator" @Alligator() -> ()
  hw.instance "cat" @Cat() -> ()
}

hw.module private @Alligator() {
  hw.instance "bear" @Bear() -> ()
}

hw.module private @Bear() {
  hw.instance "cat" @Cat() -> ()
}

hw.module private @Cat() { }

