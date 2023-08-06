; RUN: firtool --verify-diagnostics --verilog %s --lowering-options=emittedLineLength=200 | FileCheck %s
; RUN: firtool --verify-diagnostics --verilog --emit-chisel-asserts-as-sva %s --lowering-options=emittedLineLength=200 | FileCheck %s --check-prefix=SVA
; Tests extracted from:
; - test/scala/firrtl/extractverif/ExtractAssertsSpec.scala

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
    input predicate9 : UInt<1>
    input predicate10 : UInt<1>
    input enable : UInt<1>
    input other : UInt<1>
    input sum : UInt<42>

    ; CHECK: `ifndef SYNTHESIS
    ; CHECK-NEXT: always @(posedge clock) begin

    ; assert with predicate only
    ; CHECK-NEXT: if (enable & ~(predicate1 | reset)) begin
    ; CHECK-NEXT:   if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:     $error("Assertion failed (verification library): ");
    ; CHECK-NEXT:   if (`STOP_COND_)
    ; CHECK-NEXT:     $fatal;
    ; CHECK-NEXT: end

    ; SVA: wire _GEN = ~enable | predicate1 | reset;
    ; SVA-NEXT: assert__verif_library: assert property (@(posedge clock) _GEN) else $error("Assertion failed (verification library): ");

    when not(or(predicate1, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): \"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with message
    ; CHECK-NEXT: if (enable & ~(predicate2 | reset)) begin
    ; CHECK-NEXT:   if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:     $error("Assertion failed (verification library): sum =/= 1.U");
    ; CHECK-NEXT:   if (`STOP_COND_)
    ; CHECK-NEXT:     $fatal;
    ; CHECK-NEXT: end
    when not(or(predicate2, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): sum =/= 1.U\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with when
    ; CHECK-NEXT: if (other & enable & ~(predicate3 | reset)) begin
    ; CHECK-NEXT:   if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:     $error("Assertion failed (verification library): Assert with when");
    ; CHECK-NEXT:   if (`STOP_COND_)
    ; CHECK-NEXT:     $fatal;
    ; CHECK-NEXT: end
    when other :
      when not(or(predicate3, asUInt(reset))) :
        printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): Assert with when\"}</extraction-summary> bar")
        stop(clock, enable, 1)

    ; assert with message with arguments
    ; CHECK-NEXT: if (enable & ~(predicate4 | reset)) begin
    ; CHECK-NEXT:   if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:     $error("Assertion failed (verification library): expected sum === 2.U but got %d", sum);
    ; CHECK-NEXT:   if (`STOP_COND_)
    ; CHECK-NEXT:     $fatal;
    ; CHECK-NEXT: end
    when not(or(predicate4, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): expected sum === 2.U but got %d\"}</extraction-summary> bar", sum)
      stop(clock, enable, 1)

    ; CHECK-NEXT: end
    ; CHECK-NEXT: `endif


    ; assert emitted as SVA
    ; CHECK-NEXT: wire [[TMP1:.+]] = ~enable | predicate5 | reset;
    ; CHECK-NEXT: assert__verif_library: assert property (@(posedge clock) [[TMP1]])
    ; CHECK-SAME:   else $error("Assertion failed (verification library): SVA assert example, sum was %d", $sampled(sum));
    when not(or(predicate5, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): SVA assert example, sum was %d\"}</extraction-summary> bar", sum)
      stop(clock, enable, 1)

    ; assert with custom label
    ; CHECK-NEXT: wire [[TMP2:.+]] = ~enable | predicate6 | reset;
    ; CHECK-NEXT: assert__verif_library_hello_world: assert property (@(posedge clock) [[TMP2]])
    ; CHECK-SAME:   else $error("Assertion failed (verification library): Custom label example");
    when not(or(predicate6, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"format\":{\"type\":\"sva\"},\"labelExts\":[\"hello\",\"world\"],\"baseMsg\":\"Assertion failed (verification library): Custom label example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with predicate option for X-passing
    ; CHECK-NEXT: wire [[TMP3:.+]] = ~enable | predicate7 | predicate7 === 1'bx;
    ; CHECK-NEXT: assert__verif_library_0: assert property (@(posedge clock) [[TMP3]])
    ; CHECK-SAME:   else $error("Assertion failed (verification library): X-passing assert example");
    when not(predicate7) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): X-passing assert example\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; The companion assumes get bunched up in an ifdef block.
    ; CHECK-NEXT: `ifdef USE_PROPERTY_AS_CONSTRAINT
    ; CHECK-NEXT:   assume__verif_library: assume property (@(posedge clock) [[TMP1]]);
    ; CHECK-NEXT:   assume__verif_library_hello_world: assume property (@(posedge clock) [[TMP2]]);
    ; CHECK-NEXT:   assume__verif_library_0: assume property (@(posedge clock) [[TMP3]]);
    ; CHECK-NEXT: `endif


    ; assert with toggle option e.g. UNROnly
    ; CHECK-NEXT: `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   wire [[TMP4:.+]] = ~enable | predicate8 | reset;
    ; CHECK-NEXT:   assert__verif_library_1: assert property (@(posedge clock) [[TMP4]])
    ; CHECK-SAME:     else $error("Assertion failed (verification library): Conditional compilation example for UNR-only assert");
    ; CHECK-NEXT:   `ifdef USE_PROPERTY_AS_CONSTRAINT
    ; CHECK-NEXT:     always @(edge [[TMP4]])
    ; CHECK-NEXT:       assume__verif_library_1: assume([[TMP4]]);
    ; CHECK-NEXT:   `endif
    when not(or(predicate8, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"}],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): Conditional compilation example for UNR-only assert\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; if-else-fatal style assert with conditional compilation toggles
    ; CHECK-NEXT:  `ifndef SYNTHESIS
    ; CHECK-NEXT:    always @(posedge clock) begin
    ; CHECK-NEXT:     if (enable & ~(predicate9 | reset)) begin
    ; CHECK-NEXT:       if (`ASSERT_VERBOSE_COND_)
    ; CHECK-NEXT:         $error("Assertion failed (verification library): Conditional compilation example with if-else-fatal style assert");
    ; CHECK-NEXT:       if (`STOP_COND_)
    ; CHECK-NEXT:         $fatal;
    ; CHECK-NEXT:     end
    ; CHECK-NEXT:    end
    ; CHECK-NEXT:  `endif
    ; CHECK-NEXT: `endif
    when not(or(predicate9, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"}],\"format\":{\"type\":\"ifElseFatal\"},\"baseMsg\":\"Assertion failed (verification library): Conditional compilation example with if-else-fatal style assert\"}</extraction-summary> bar")
      stop(clock, enable, 1)

    ; assert with multiple toggle options
    ; CHECK-NEXT: `ifdef USE_FORMAL_ONLY_CONSTRAINTS
    ; CHECK-NEXT:   `ifdef USE_UNR_ONLY_CONSTRAINTS
    ; CHECK-NEXT:     wire [[TMP5:.+]] = ~enable | predicate10 | reset;
    ; CHECK-NEXT:     assert__verif_library_2: assert property (@(posedge clock) [[TMP5]])
    ; CHECK-SAME:       else $error("Assertion failed (verification library): Conditional compilation example for UNR-only and formal-only assert");
    ; CHECK-NEXT:     `ifdef USE_PROPERTY_AS_CONSTRAINT
    ; CHECK-NEXT:       always @(edge [[TMP5]])
    ; CHECK-NEXT:         assume__verif_library_2: assume([[TMP5]]);
    ; CHECK-NEXT:     `endif
    ; CHECK-NEXT:   `endif
    ; CHECK-NEXT: `endif
    when not(or(predicate10, asUInt(reset))) :
      printf(clock, enable, "foo [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"formalOnly\"},{\"type\":\"unrOnly\"}],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Assertion failed (verification library): Conditional compilation example for UNR-only and formal-only assert\"}</extraction-summary> bar")
      stop(clock, enable, 1)
