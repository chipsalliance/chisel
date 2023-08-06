; RUN: firtool --verify-diagnostics --verilog %s --lowering-options=emittedLineLength=500 | FileCheck %s
; Tests extracted from:
; - test/scala/firrtl/extractverif/ExtractAssumesSpec.scala

circuit Foo:
  module Foo:
    input clock : Clock
    input reset : AsyncReset
    input predicate1 : UInt<1>
    input predicate2 : UInt<1>
    input predicate3 : UInt<1>
    input predicate4 : UInt<1>
    input predicate5 : UInt<1>
    input predicate6 : UInt<1>
    input predicate7 : UInt<1>
    input predicate8 : UInt<1>
    input enable : UInt<1>
    input other : UInt<1>
    input sum : UInt<42>

    ; assume with predicate only
    ; CHECK: {{assume__verif_library.*}}: assume property (@(posedge clock) ~enable | predicate1 | reset)
    ; CHECK-SAME:   else $error("Assumption does not hold (verification library): ");
    when not(or(predicate1, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"baseMsg\":\"Assumption does not hold (verification library): \"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with message
    ; CHECK-NEXT: {{assume__verif_library.*}}: assume property (@(posedge clock) ~enable | predicate2 | reset)
    ; CHECK-SAME:   else $error("Assumption does not hold (verification library): sum =/= 1.U");
    when not(or(predicate2, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"baseMsg\":\"Assumption does not hold (verification library): sum =/= 1.U\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with when
    ; CHECK-NEXT: {{assume__verif_library.*}}: assume property (@(posedge clock) ~(other & enable) | predicate3 | reset)
    ; CHECK-SAME:   else $error("Assumption does not hold (verification library): assume with when");
    when other :
      when not(or(predicate3, asUInt(reset))) :
        printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"baseMsg\":\"Assumption does not hold (verification library): assume with when\"}</extraction-summary> bar")
        stop(clock, enable, 1)

    ; assume with message with arguments
    ; CHECK-NEXT: {{assume__verif_library.*}}: assume property (@(posedge clock) ~enable | predicate4 | reset)
    ; CHECK-SAME:   else $error("Assumption does not hold (verification library): expected sum === 2.U but got %d", $sampled(sum));
    when not(or(predicate4, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"baseMsg\":\"Assumption does not hold (verification library): expected sum === 2.U but got %d\"}</extraction-summary> bar", sum)
      stop(clock, enable, 1)

    ; assume with custom label
    ; CHECK-NEXT: assume__verif_library_hello_world: assume property (@(posedge clock) ~enable | predicate5 | reset)
    ; CHECK-SAME:   else $error("Assumption does not hold (verification library): Custom label example");
    when not(or(predicate5, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"labelExts\":[\"hello\",\"world\"],\"baseMsg\":\"Assumption does not hold (verification library): Custom label example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with predicate option for X-passing
    ; CHECK-NEXT: {{assume__verif_library.*}}: assume property (@(posedge clock) ~enable | predicate6 | predicate6 === 1'bx)
    ; CHECK-SAME:   else $error("Assumption does not hold (verification library): X-passing assume example");
    when not(predicate6) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"baseMsg\":\"Assumption does not hold (verification library): X-passing assume example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with toggle option e.g. UNROnly
    ; CHECK-NEXT: `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   {{assume__verif_library.*}}: assume property (@(posedge clock) ~enable | predicate7 | reset)
    ; CHECK-SAME:     else $error("Assumption does not hold (verification library): Conditional compilation example for UNR-only assume");
    ; CHECK-NEXT: `endif
    when not(or(predicate7, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"}],\"baseMsg\":\"Assumption does not hold (verification library): Conditional compilation example for UNR-only assume\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assume with multiple toggle options
    ; CHECK-NEXT: `ifdef USE_FORMAL_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:     {{assume__verif_library.*}}: assume property (@(posedge clock) ~enable | predicate8 | reset)
    ; CHECK-SAME:       else $error("Assumption does not hold (verification library): Conditional compilation example for UNR-only and formal-only assume");
    ; CHECK-NEXT:   `endif
    ; CHECK-NEXT: `endif
    when not(or(predicate8, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"formalOnly\"},{\"type\":\"unrOnly\"}],\"baseMsg\":\"Assumption does not hold (verification library): Conditional compilation example for UNR-only and formal-only assume\"}</extraction-summary> bar")
      stop(clock, enable, 1)
