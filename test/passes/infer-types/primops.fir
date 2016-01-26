; RUN: firrtl -i %s -o %s.v -X verilog -p ct 2>&1 | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module top :
      input clk : Clock
      wire a : UInt<16>
      wire b : UInt<8>
      wire c : SInt<16>
      wire d : SInt<8>
      wire e : UInt<1>

      a <= UInt(1)
      b <= UInt(1)
      c <= SInt(1)
      d <= SInt(1)
      e <= UInt(1)

      node vadd = add(a, c)    ;CHECK: node vadd = add(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wadd = add(a, b) ;CHECK: node wadd = add(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xadd = add(a, d) ;CHECK: node xadd = add(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node yadd = add(c, b) ;CHECK: node yadd = add(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zadd = add(c, d) ;CHECK: node zadd = add(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vsub = sub(a, c)    ;CHECK: node vsub = sub(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wsub = sub(a, b) ;CHECK: node wsub = sub(a@<t:UInt>, b@<t:UInt>)@<t:SInt>
      node xsub = sub(a, d) ;CHECK: node xsub = sub(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node ysub = sub(c, b) ;CHECK: node ysub = sub(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zsub = sub(c, d) ;CHECK: node zsub = sub(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vmul = mul(a, c)    ;CHECK: node vmul = mul(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wmul = mul(a, b) ;CHECK: node wmul = mul(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xmul = mul(a, d) ;CHECK: node xmul = mul(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node ymul = mul(c, b) ;CHECK: node ymul = mul(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zmul = mul(c, d) ;CHECK: node zmul = mul(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vdiv = div(a, c)    ;CHECK: node vdiv = div(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wdiv = div(a, b) ;CHECK: node wdiv = div(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xdiv = div(a, d) ;CHECK: node xdiv = div(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node ydiv = div(c, b) ;CHECK: node ydiv = div(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zdiv = div(c, d) ;CHECK: node zdiv = div(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vmod = mod(a, c)    ;CHECK: node vmod = mod(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wmod = mod(a, b) ;CHECK: node wmod = mod(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xmod = mod(a, d) ;CHECK: node xmod = mod(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node ymod = mod(c, b) ;CHECK: node ymod = mod(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zmod = mod(c, d) ;CHECK: node zmod = mod(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vlt = lt(a, c)    ;CHECK: node vlt = lt(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wlt = lt(a, b) ;CHECK: node wlt = lt(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xlt = lt(a, d) ;CHECK: node xlt = lt(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node ylt = lt(c, b) ;CHECK: node ylt = lt(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zlt = lt(c, d) ;CHECK: node zlt = lt(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vleq = leq(a, c)    ;CHECK: node vleq = leq(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wleq = leq(a, b) ;CHECK: node wleq = leq(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xleq = leq(a, d) ;CHECK: node xleq = leq(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node yleq = leq(c, b) ;CHECK: node yleq = leq(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zleq = leq(c, d) ;CHECK: node zleq = leq(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vgt = gt(a, c)    ;CHECK: node vgt = gt(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wgt = gt(a, b) ;CHECK: node wgt = gt(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xgt = gt(a, d) ;CHECK: node xgt = gt(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node ygt = gt(c, b) ;CHECK: node ygt = gt(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zgt = gt(c, d) ;CHECK: node zgt = gt(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vgeq = geq(a, c)    ;CHECK: node vgeq = geq(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wgeq = geq(a, b) ;CHECK: node wgeq = geq(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xgeq = geq(a, d) ;CHECK: node xgeq = geq(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node ygeq = geq(c, b) ;CHECK: node ygeq = geq(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zgeq = geq(c, d) ;CHECK: node zgeq = geq(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node veq = eq(a, c)    ;CHECK: node veq = eq(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node weq = eq(a, b) ;CHECK: node weq = eq(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xeq = eq(a, d) ;CHECK: node xeq = eq(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node yeq = eq(c, b) ;CHECK: node yeq = eq(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zeq = eq(c, d) ;CHECK: node zeq = eq(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vneq = neq(a, c)    ;CHECK: node vneq = neq(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wneq = neq(a, b) ;CHECK: node wneq = neq(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xneq = neq(a, d) ;CHECK: node xneq = neq(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node yneq = neq(c, b) ;CHECK: node yneq = neq(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zneq = neq(c, d) ;CHECK: node zneq = neq(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vpad = pad(a, 10)   ;CHECK: node vpad = pad(a@<t:UInt>, 10)@<t:UInt>
      node wpad = pad(a, 10) ;CHECK: node wpad = pad(a@<t:UInt>, 10)@<t:UInt>
      node zpad = pad(c, 10) ;CHECK: node zpad = pad(c@<t:SInt>, 10)@<t:SInt>
   
      node vasUInt = asUInt(d) ;CHECK: node vasUInt = asUInt(d@<t:SInt>)@<t:UInt>
      node wasUInt = asUInt(a) ;CHECK: node wasUInt = asUInt(a@<t:UInt>)@<t:UInt>
      node zasUInt = asUInt(clk) ;CHECK: node zasUInt = asUInt(clk@<t:Clock>)@<t:UInt>
   
      node vasSInt = asSInt(a) ;CHECK: node vasSInt = asSInt(a@<t:UInt>)@<t:SInt>
      node wasSInt = asSInt(c) ;CHECK: node wasSInt = asSInt(c@<t:SInt>)@<t:SInt>
      node zasSInt = asSInt(clk) ;CHECK: node zasSInt = asSInt(clk@<t:Clock>)@<t:SInt>
   
      node vasClock = asClock(a) ;CHECK: node vasClock = asClock(a@<t:UInt>)@<t:Clock>
      node wasClock = asClock(c) ;CHECK: node wasClock = asClock(c@<t:SInt>)@<t:Clock>
      node zasClock = asClock(clk) ;CHECK: node zasClock = asClock(clk@<t:Clock>)@<t:Clock>
   
      node vshl = shl(a, 10)   ;CHECK: node vshl = shl(a@<t:UInt>, 10)@<t:UInt>
      node wshl = shl(a, 10) ;CHECK: node wshl = shl(a@<t:UInt>, 10)@<t:UInt>
      node zshl = shl(c, 10) ;CHECK: node zshl = shl(c@<t:SInt>, 10)@<t:SInt>
   
      node vdshl = dshl(a, a)   ;CHECK: node vdshl = dshl(a@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node wdshl = dshl(a, a) ;CHECK: node wdshl = dshl(a@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node zdshl = dshl(c, a) ;CHECK: node zdshl = dshl(c@<t:SInt>, a@<t:UInt>)@<t:SInt>
   
      node vdshr = dshr(a, a)   ;CHECK: node vdshr = dshr(a@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node wdshr = dshr(a, a) ;CHECK: node wdshr = dshr(a@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node zdshr = dshr(c, a) ;CHECK: node zdshr = dshr(c@<t:SInt>, a@<t:UInt>)@<t:SInt>
   
      node vshr = shr(a, 10)   ;CHECK: node vshr = shr(a@<t:UInt>, 10)@<t:UInt>
      node wshr = shr(a, 10) ;CHECK: node wshr = shr(a@<t:UInt>, 10)@<t:UInt>
      node zshr = shr(c, 10) ;CHECK: node zshr = shr(c@<t:SInt>, 10)@<t:SInt>
   
      node vcvt = cvt(a)   ;CHECK: node vcvt = cvt(a@<t:UInt>)@<t:SInt>
      node wcvt = cvt(a) ;CHECK: node wcvt = cvt(a@<t:UInt>)@<t:SInt>
      node zcvt = cvt(c) ;CHECK: node zcvt = cvt(c@<t:SInt>)@<t:SInt>
   
      node vneg = neg(a)   ;CHECK: node vneg = neg(a@<t:UInt>)@<t:SInt>
      node wneg = neg(a) ;CHECK: node wneg = neg(a@<t:UInt>)@<t:SInt>
      node zneg = neg(c) ;CHECK: node zneg = neg(c@<t:SInt>)@<t:SInt>
   
      node wnot = not(a) ;CHECK: node wnot = not(a@<t:UInt>)@<t:UInt>
      node unot = not(c) ;CHECK: node unot = not(c@<t:SInt>)@<t:UInt>

      node WAND = and(a, b) ;CHECK: node WAND = and(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node uand = and(c, d) ;CHECK: node uand = and(c@<t:SInt>, d@<t:SInt>)@<t:UInt>

      node WOR = or(a, b)  ;CHECK: node WOR = or(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node uor = or(c, d)  ;CHECK: node uor = or(c@<t:SInt>, d@<t:SInt>)@<t:UInt>

      node wxor = xor(a, b) ;CHECK: node wxor = xor(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node uxor = xor(c, d) ;CHECK: node uxor = xor(c@<t:SInt>, d@<t:SInt>)@<t:UInt>

      node wbits = bits(a, 2, 0) ;CHECK: node wbits = bits(a@<t:UInt>, 2, 0)@<t:UInt>
      node ubits = bits(c, 2, 0) ;CHECK: node ubits = bits(c@<t:SInt>, 2, 0)@<t:UInt>

      node xcat = cat(a, b) ;CHECK: node xcat = cat(a@<t:UInt>, b@<t:UInt>)@<t:UInt>

      node uandr = andr(a, b, a) ;CHECK: node uandr = andr(a@<t:UInt>, b@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node uorr = orr(a, b, a) ;CHECK: node uorr = orr(a@<t:UInt>, b@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node uxorr = xorr(a, b, a) ;CHECK: node uxorr = xorr(a@<t:UInt>, b@<t:UInt>, a@<t:UInt>)@<t:UInt>

      node whead = head(a, 2) ;CHECK: node whead = head(a@<t:UInt>, 2)@<t:UInt>
      node uhead = head(c, 2) ;CHECK: node uhead = head(c@<t:SInt>, 2)@<t:UInt>

      node wtail = tail(a, 2) ;CHECK: node wtail = tail(a@<t:UInt>, 2)@<t:UInt>
      node utail = tail(c, 2) ;CHECK: node utail = tail(c@<t:SInt>, 2)@<t:UInt>

;CHECK: Finished Infer Types
