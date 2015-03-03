; RUN: firrtl %s abcde ct | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module top :
      wire a : UInt(16)
      wire b : UInt(8)
      wire c : SInt(16)
      wire d : SInt(8)

      ;add
      module add :
         node w = adduu(a,b) ;CHECK: node w = adduu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = addus(a,d) ;CHECK: node x = addus(a@<t:UInt>,d@<t:SInt>)@<t:SInt>
         node y = addsu(c,b) ;CHECK: node y = addsu(c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = addss(c,d) ;CHECK: node z = addss(c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;sub
      module sub :
         node w = subuu(a,b) ;CHECK: node w = subuu(a@<t:UInt>,b@<t:UInt>)@<t:SInt>
         node x = subus(a,d) ;CHECK: node x = subus(a@<t:UInt>,d@<t:SInt>)@<t:SInt>
         node y = subsu(c,b) ;CHECK: node y = subsu(c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = subss(c,d) ;CHECK: node z = subss(c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;mul
      module mul :
         node w = muluu(a,b) ;CHECK: node w = muluu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = mulus(a,d) ;CHECK: node x = mulus(a@<t:UInt>,d@<t:SInt>)@<t:SInt>
         node y = mulsu(c,b) ;CHECK: node y = mulsu(c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = mulss(c,d) ;CHECK: node z = mulss(c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;div
      module div :
         node w = divuu(a,b) ;CHECK: node w = divuu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = divus(a,d) ;CHECK: node x = divus(a@<t:UInt>,d@<t:SInt>)@<t:SInt>
         node y = divsu(c,b) ;CHECK: node y = divsu(c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = divss(c,d) ;CHECK: node z = divss(c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;mod
      module mod :
         node w = moduu(a,b) ;CHECK: node w = moduu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = modus(a,d) ;CHECK: node x = modus(a@<t:UInt>,d@<t:SInt>)@<t:SInt>
         node y = modsu(c,b) ;CHECK: node y = modsu(c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = modss(c,d) ;CHECK: node z = modss(c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;rem
      module rem :
         node w = remuu(a,b) ;CHECK: node w = remuu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = remus(a,d) ;CHECK: node x = remus(a@<t:UInt>,d@<t:SInt>)@<t:SInt>
         node y = remsu(c,b) ;CHECK: node y = remsu(c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = remss(c,d) ;CHECK: node z = remss(c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;add-mod
      module add-mod :
         node w = add-moduu(a,b) ;CHECK: node w = add-moduu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = add-modus(a,d) ;CHECK: node x = add-modus(a@<t:UInt>,d@<t:SInt>)@<t:UInt>
         node y = add-modsu(c,b) ;CHECK: node y = add-modsu(c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = add-modss(c,d) ;CHECK: node z = add-modss(c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;sub-mod
      module sub-mod :
         node w = sub-moduu(a,b) ;CHECK: node w = sub-moduu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = sub-modus(a,d) ;CHECK: node x = sub-modus(a@<t:UInt>,d@<t:SInt>)@<t:UInt>
         node y = sub-modsu(c,b) ;CHECK: node y = sub-modsu(c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = sub-modss(c,d) ;CHECK: node z = sub-modss(c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;lt
      module lt :
         node w = ltuu(a,b) ;CHECK: node w = ltuu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = ltus(a,d) ;CHECK: node x = ltus(a@<t:UInt>,d@<t:SInt>)@<t:UInt>
         node y = ltsu(c,b) ;CHECK: node y = ltsu(c@<t:SInt>,b@<t:UInt>)@<t:UInt>
         node z = ltss(c,d) ;CHECK: node z = ltss(c@<t:SInt>,d@<t:SInt>)@<t:UInt>
      ;leq
      module leq :
         node w = lequu(a,b) ;CHECK: node w = lequu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = lequs(a,d) ;CHECK: node x = lequs(a@<t:UInt>,d@<t:SInt>)@<t:UInt>
         node y = leqsu(c,b) ;CHECK: node y = leqsu(c@<t:SInt>,b@<t:UInt>)@<t:UInt>
         node z = leqss(c,d) ;CHECK: node z = leqss(c@<t:SInt>,d@<t:SInt>)@<t:UInt>
      ;gt
      module gt :
         node w = gtuu(a,b) ;CHECK: node w = gtuu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = gtus(a,d) ;CHECK: node x = gtus(a@<t:UInt>,d@<t:SInt>)@<t:UInt>
         node y = gtsu(c,b) ;CHECK: node y = gtsu(c@<t:SInt>,b@<t:UInt>)@<t:UInt>
         node z = gtss(c,d) ;CHECK: node z = gtss(c@<t:SInt>,d@<t:SInt>)@<t:UInt>
      ;geq
      module geq :
         node w = gequu(a,b) ;CHECK: node w = gequu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = gequs(a,d) ;CHECK: node x = gequs(a@<t:UInt>,d@<t:SInt>)@<t:UInt>
         node y = geqsu(c,b) ;CHECK: node y = geqsu(c@<t:SInt>,b@<t:UInt>)@<t:UInt>
         node z = geqss(c,d) ;CHECK: node z = geqss(c@<t:SInt>,d@<t:SInt>)@<t:UInt>
      ;pad
      module pad :
         node w = paduu(a,b) ;CHECK: node w = paduu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = padus(a,d) ;CHECK: node x = padus(a@<t:UInt>,d@<t:SInt>)@<t:UInt>
         node y = padsu(c,b) ;CHECK: node y = padsu(c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = padss(c,d) ;CHECK: node z = padss(c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;and
      module and :
         node w = and(a,b) ;CHECK: node w = and(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
      module or :
         node w = or(a,b)  ;CHECK: node w = or(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
      module xor :
         node w = xor(a,b) ;CHECK: node w = xor(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
      ;concat
         node w = concat(a,b) ;CHECK: node w = concat(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
      ;equal
         node w = equaluu(a,b) ;CHECK: node w = equaluu(a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = equalus(a,d) ;CHECK: node x = equalus(a@<t:UInt>,d@<t:SInt>)@<t:UInt>
         node y = equalsu(c,b) ;CHECK: node y = equalsu(c@<t:SInt>,b@<t:UInt>)@<t:UInt>
         node z = equalss(c,d) ;CHECK: node z = equalss(c@<t:SInt>,d@<t:SInt>)@<t:UInt>
      ;mux
         node w = muxuu(e,a,b) ;CHECK: node w = muxuu(e@<t:UInt>,a@<t:UInt>,b@<t:UInt>)@<t:UInt>
         node x = muxus(e,a,d) ;CHECK: node x = muxus(e@<t:UInt>,a@<t:UInt>,d@<t:SInt>)@<t:SInt>
         node y = muxsu(e,c,b) ;CHECK: node y = muxsu(e@<t:UInt>,c@<t:SInt>,b@<t:UInt>)@<t:SInt>
         node z = muxss(e,c,d) ;CHECK: node z = muxss(e@<t:UInt>,c@<t:SInt>,d@<t:SInt>)@<t:SInt>
      ;shl
      ;shr
      ;bit
      ;bits




;CHECK: Finished Infer Types
