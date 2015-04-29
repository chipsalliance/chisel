; RUN: firrtl -i %s -o %s.flo -x abcde -p ct | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module top :
      wire a : UInt<16>
      wire b : UInt<8>
      wire c : SInt<16>
      wire d : SInt<8>
      wire e : UInt<1>

      node vadd = add(a, c)    ;CHECK: node vadd = add-us(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wadd-uu = add-uu(a, b) ;CHECK: node wadd-uu = add-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xadd-us = add-us(a, d) ;CHECK: node xadd-us = add-us(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node yadd-su = add-su(c, b) ;CHECK: node yadd-su = add-su(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zadd-ss = add-ss(c, d) ;CHECK: node zadd-ss = add-ss(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vsub = sub(a, c)    ;CHECK: node vsub = sub-us(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wsub-uu = sub-uu(a, b) ;CHECK: node wsub-uu = sub-uu(a@<t:UInt>, b@<t:UInt>)@<t:SInt>
      node xsub-us = sub-us(a, d) ;CHECK: node xsub-us = sub-us(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node ysub-su = sub-su(c, b) ;CHECK: node ysub-su = sub-su(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zsub-ss = sub-ss(c, d) ;CHECK: node zsub-ss = sub-ss(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vmul = mul(a, c)    ;CHECK: node vmul = mul-us(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wmul-uu = mul-uu(a, b) ;CHECK: node wmul-uu = mul-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xmul-us = mul-us(a, d) ;CHECK: node xmul-us = mul-us(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node ymul-su = mul-su(c, b) ;CHECK: node ymul-su = mul-su(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zmul-ss = mul-ss(c, d) ;CHECK: node zmul-ss = mul-ss(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vdiv = div(a, c)    ;CHECK: node vdiv = div-us(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wdiv-uu = div-uu(a, b) ;CHECK: node wdiv-uu = div-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xdiv-us = div-us(a, d) ;CHECK: node xdiv-us = div-us(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node ydiv-su = div-su(c, b) ;CHECK: node ydiv-su = div-su(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zdiv-ss = div-ss(c, d) ;CHECK: node zdiv-ss = div-ss(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vmod = mod(a, c)    ;CHECK: node vmod = mod-us(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wmod-uu = mod-uu(a, b) ;CHECK: node wmod-uu = mod-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xmod-us = mod-us(a, d) ;CHECK: node xmod-us = mod-us(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node ymod-su = mod-su(c, b) ;CHECK: node ymod-su = mod-su(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zmod-ss = mod-ss(c, d) ;CHECK: node zmod-ss = mod-ss(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vquo = quo(a, c)    ;CHECK: node vquo = quo-us(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wquo-uu = quo-uu(a, b) ;CHECK: node wquo-uu = quo-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xquo-us = quo-us(a, d) ;CHECK: node xquo-us = quo-us(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node yquo-su = quo-su(c, b) ;CHECK: node yquo-su = quo-su(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zquo-ss = quo-ss(c, d) ;CHECK: node zquo-ss = quo-ss(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vrem = rem(a, c)    ;CHECK: node vrem = rem-us(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wrem-uu = rem-uu(a, b) ;CHECK: node wrem-uu = rem-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xrem-us = rem-us(a, d) ;CHECK: node xrem-us = rem-us(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node yrem-su = rem-su(c, b) ;CHECK: node yrem-su = rem-su(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zrem-ss = rem-ss(c, d) ;CHECK: node zrem-ss = rem-ss(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vadd-wrap = add-wrap(a, c)    ;CHECK: node vadd-wrap = add-wrap-us(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wadd-wrap-uu = add-wrap-uu(a, b) ;CHECK: node wadd-wrap-uu = add-wrap-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xadd-wrap-us = add-wrap-us(a, d) ;CHECK: node xadd-wrap-us = add-wrap-us(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node yadd-wrap-su = add-wrap-su(c, b) ;CHECK: node yadd-wrap-su = add-wrap-su(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zadd-wrap-ss = add-wrap-ss(c, d) ;CHECK: node zadd-wrap-ss = add-wrap-ss(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vsub-wrap = sub-wrap(a, c)    ;CHECK: node vsub-wrap = sub-wrap-us(a@<t:UInt>, c@<t:SInt>)@<t:SInt>
      node wsub-wrap-uu = sub-wrap-uu(a, b) ;CHECK: node wsub-wrap-uu = sub-wrap-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xsub-wrap-us = sub-wrap-us(a, d) ;CHECK: node xsub-wrap-us = sub-wrap-us(a@<t:UInt>, d@<t:SInt>)@<t:SInt>
      node ysub-wrap-su = sub-wrap-su(c, b) ;CHECK: node ysub-wrap-su = sub-wrap-su(c@<t:SInt>, b@<t:UInt>)@<t:SInt>
      node zsub-wrap-ss = sub-wrap-ss(c, d) ;CHECK: node zsub-wrap-ss = sub-wrap-ss(c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vlt = lt(a, c)    ;CHECK: node vlt = lt-us(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wlt-uu = lt-uu(a, b) ;CHECK: node wlt-uu = lt-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xlt-us = lt-us(a, d) ;CHECK: node xlt-us = lt-us(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node ylt-su = lt-su(c, b) ;CHECK: node ylt-su = lt-su(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zlt-ss = lt-ss(c, d) ;CHECK: node zlt-ss = lt-ss(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vleq = leq(a, c)    ;CHECK: node vleq = leq-us(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wleq-uu = leq-uu(a, b) ;CHECK: node wleq-uu = leq-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xleq-us = leq-us(a, d) ;CHECK: node xleq-us = leq-us(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node yleq-su = leq-su(c, b) ;CHECK: node yleq-su = leq-su(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zleq-ss = leq-ss(c, d) ;CHECK: node zleq-ss = leq-ss(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vgt = gt(a, c)    ;CHECK: node vgt = gt-us(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wgt-uu = gt-uu(a, b) ;CHECK: node wgt-uu = gt-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xgt-us = gt-us(a, d) ;CHECK: node xgt-us = gt-us(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node ygt-su = gt-su(c, b) ;CHECK: node ygt-su = gt-su(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zgt-ss = gt-ss(c, d) ;CHECK: node zgt-ss = gt-ss(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vgeq = geq(a, c)    ;CHECK: node vgeq = geq-us(a@<t:UInt>, c@<t:SInt>)@<t:UInt>
      node wgeq-uu = geq-uu(a, b) ;CHECK: node wgeq-uu = geq-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xgeq-us = geq-us(a, d) ;CHECK: node xgeq-us = geq-us(a@<t:UInt>, d@<t:SInt>)@<t:UInt>
      node ygeq-su = geq-su(c, b) ;CHECK: node ygeq-su = geq-su(c@<t:SInt>, b@<t:UInt>)@<t:UInt>
      node zgeq-ss = geq-ss(c, d) ;CHECK: node zgeq-ss = geq-ss(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vneq = neq(a, b)    ;CHECK: node vneq = neq-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node wneq-uu = neq-uu(a, b) ;CHECK: node wneq-uu = neq-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node zneq-ss = neq-ss(c, d) ;CHECK: node zneq-ss = neq-ss(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node veq = eq(a, b)    ;CHECK: node veq = eq-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node weq-uu = eq-uu(a, b) ;CHECK: node weq-uu = eq-uu(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node zeq-ss = eq-ss(c, d) ;CHECK: node zeq-ss = eq-ss(c@<t:SInt>, d@<t:SInt>)@<t:UInt>
   
      node vmux = mux(e, a, b)    ;CHECK: node vmux = mux-uu(e@<t:UInt>, a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node wmux-uu = mux-uu(e, a, b) ;CHECK: node wmux-uu = mux-uu(e@<t:UInt>, a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node zmux-ss = mux-ss(e, c, d) ;CHECK: node zmux-ss = mux-ss(e@<t:UInt>, c@<t:SInt>, d@<t:SInt>)@<t:SInt>
   
      node vpad = pad(a, 10)   ;CHECK: node vpad = pad-u(a@<t:UInt>, 10)@<t:UInt>
      node wpad-u = pad-u(a, 10) ;CHECK: node wpad-u = pad-u(a@<t:UInt>, 10)@<t:UInt>
      node zpad-s = pad-s(c, 10) ;CHECK: node zpad-s = pad-s(c@<t:SInt>, 10)@<t:SInt>
   
      node vas-UInt = as-UInt(d)   ;CHECK: node vas-UInt = as-UInt-s(d@<t:SInt>)@<t:UInt>
      node was-UInt-u = as-UInt-u(a) ;CHECK: node was-UInt-u = as-UInt-u(a@<t:UInt>)@<t:UInt>
      node zas-UInt-s = as-UInt-s(c) ;CHECK: node zas-UInt-s = as-UInt-s(c@<t:SInt>)@<t:UInt>
   
      node vas-SInt = as-SInt(a)   ;CHECK: node vas-SInt = as-SInt-u(a@<t:UInt>)@<t:SInt>
      node was-SInt-u = as-SInt-u(a) ;CHECK: node was-SInt-u = as-SInt-u(a@<t:UInt>)@<t:SInt>
      node zas-SInt-s = as-SInt-s(c) ;CHECK: node zas-SInt-s = as-SInt-s(c@<t:SInt>)@<t:SInt>
   
      node vshl = shl(a, 10)   ;CHECK: node vshl = shl-u(a@<t:UInt>, 10)@<t:UInt>
      node wshl-u = shl-u(a, 10) ;CHECK: node wshl-u = shl-u(a@<t:UInt>, 10)@<t:UInt>
      node zshl-s = shl-s(c, 10) ;CHECK: node zshl-s = shl-s(c@<t:SInt>, 10)@<t:SInt>
   
      node vdshl = dshl(a, a)   ;CHECK: node vdshl = dshl-u(a@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node wdshl-u = dshl-u(a, a) ;CHECK: node wdshl-u = dshl-u(a@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node zdshl-s = dshl-s(c, a) ;CHECK: node zdshl-s = dshl-s(c@<t:SInt>, a@<t:UInt>)@<t:SInt>
   
      node vdshr = dshr(a, a)   ;CHECK: node vdshr = dshr-u(a@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node wdshr-u = dshr-u(a, a) ;CHECK: node wdshr-u = dshr-u(a@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node zdshr-s = dshr-s(c, a) ;CHECK: node zdshr-s = dshr-s(c@<t:SInt>, a@<t:UInt>)@<t:SInt>
   
      node vshr = shr(a, 10)   ;CHECK: node vshr = shr-u(a@<t:UInt>, 10)@<t:UInt>
      node wshr-u = shr-u(a, 10) ;CHECK: node wshr-u = shr-u(a@<t:UInt>, 10)@<t:UInt>
      node zshr-s = shr-s(c, 10) ;CHECK: node zshr-s = shr-s(c@<t:SInt>, 10)@<t:SInt>
   
      node vconvert = convert(a)   ;CHECK: node vconvert = convert-u(a@<t:UInt>)@<t:SInt>
      node wconvert-u = convert-u(a) ;CHECK: node wconvert-u = convert-u(a@<t:UInt>)@<t:SInt>
      node zconvert-s = convert-s(c) ;CHECK: node zconvert-s = convert-s(c@<t:SInt>)@<t:SInt>
   
      node vneg = neg(a)   ;CHECK: node vneg = neg-u(a@<t:UInt>)@<t:UInt>
      node wneg-u = neg-u(a) ;CHECK: node wneg-u = neg-u(a@<t:UInt>)@<t:UInt>
      node zneg-s = neg-s(c) ;CHECK: node zneg-s = neg-s(c@<t:SInt>)@<t:UInt>
   
      node uand = bit-and(a, b) ;CHECK: node uand = bit-and(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node vor = bit-or(a, b)  ;CHECK: node vor = bit-or(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node wxor = bit-xor(a, b) ;CHECK: node wxor = bit-xor(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node xcat = cat(a, b) ;CHECK: node xcat = cat(a@<t:UInt>, b@<t:UInt>)@<t:UInt>
      node ybit = bit(a, 0) ;CHECK: node ybit = bit(a@<t:UInt>, 0)@<t:UInt>
      node zbits = bits(a, 2, 0) ;CHECK: node zbits = bits(a@<t:UInt>, 2, 0)@<t:UInt>

      node uand-reduce = bit-and-reduce(a, b, a) ;CHECK: node uand-reduce = bit-and-reduce(a@<t:UInt>, b@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node uor-reduce = bit-or-reduce(a, b, a) ;CHECK: node uor-reduce = bit-or-reduce(a@<t:UInt>, b@<t:UInt>, a@<t:UInt>)@<t:UInt>
      node uxor-reduce = bit-xor-reduce(a, b, a) ;CHECK: node uxor-reduce = bit-xor-reduce(a@<t:UInt>, b@<t:UInt>, a@<t:UInt>)@<t:UInt>

;CHECK: Finished Infer Types
