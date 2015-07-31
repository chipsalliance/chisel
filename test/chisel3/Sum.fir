; RUN: firrtl -i %s -o %s.v -X verilog -p c | tee %s.out | FileCheck %s
;CHECK: Done!

circuit SumTop :
   module Sum :
      input sreq : {valid : UInt<1>, bits : {v : UInt<32>, len : UInt<32>}, flip ready : UInt<1>}
      output srsp : {valid : UInt<1>, bits : {data : UInt<32>}, flip ready : UInt<1>}
      input mrsp : {valid : UInt<1>, bits : {tag : UInt<8>, data : UInt<32>}, flip ready : UInt<1>}
      output mreq : {valid : UInt<1>, bits : {wr? : UInt<1>, tag : UInt<8>, addr : UInt<32>, data : UInt<32>}, flip ready : UInt<1>}
      reg a2 : UInt<32>
      reg ea3 : UInt<32>
      reg sum4 : UInt<32>
      reg num_reqs5 : UInt<?>
      on-reset num_reqs5 := UInt<8>(0)
      reg computing?6 : UInt<?>
      on-reset computing?6 := UInt<1>(0)
      mrsp.ready := UInt<1>(1)
      sreq.ready := UInt<1>(1)
      srsp.valid := UInt<1>(0)
      srsp.bits.data := UInt<1>(0)
      mrsp.ready := UInt<1>(1)
      mreq.bits.wr? := UInt<1>(0)
      mreq.bits.data := UInt<1>(0)
      mreq.bits.addr := UInt<1>(0)
      mreq.valid := UInt<1>(0)
      mreq.bits.tag := UInt<1>(0)
      when computing?6 :
         sreq.ready := UInt<1>(0)
         when mrsp.valid :
            node tmp7 = add-wrap(sum4, mrsp.bits.data)
            sum4 := tmp7         
         node tmp8 = lt(a2, ea3)
         when tmp8 :
            mreq.valid := UInt<1>(1)
            mreq.bits.addr := a2
            when mreq.ready :
               node tmp9 = add-wrap(a2, UInt<1>(1))
               a2 := tmp9         
         else :            
            node tmp10 = eq(num_reqs5, UInt<1>(0))
            when tmp10 :
               srsp.valid := UInt<1>(1)
               srsp.bits.data := sum4
               when srsp.ready :
                  sum4 := UInt<1>(0)
                  computing?6 := UInt<1>(0)      
      else :
         when sreq.valid :
            a2 := sreq.bits.v
            node tmp11 = add-wrap(sreq.bits.v, sreq.bits.len)
            ea3 := tmp11   
   module Memory :
      input req : {valid : UInt<1>, bits : {wr? : UInt<1>, tag : UInt<8>, addr : UInt<32>, data : UInt<32>}, flip ready : UInt<1>}
      output rsp : {valid : UInt<1>, bits : {tag : UInt<8>, data : UInt<32>}, flip ready : UInt<1>}
      cmem mem13 : UInt<32>[8]
      req.ready := rsp.ready
      rsp.valid := UInt<1>(0)
      rsp.bits.data := UInt<1>(0)
      rsp.bits.tag := UInt<1>(0)
      when req.valid :
         rsp.valid := UInt<1>(1)
         rsp.bits.tag := req.bits.tag
         when req.bits.wr? :
            rsp.bits.data := req.bits.data
            accessor a14 = mem13[req.bits.addr]
            a14 := req.bits.data         
         else :
            accessor a15 = mem13[req.bits.addr]
            rsp.bits.data := a15   
   module ArbiterRR :
      input in : {valid : UInt<1>, bits : {wr? : UInt<1>, tag : UInt<8>, addr : UInt<32>, data : UInt<32>}, flip ready : UInt<1>}[2]
      output out : {valid : UInt<1>, bits : {wr? : UInt<1>, tag : UInt<8>, addr : UInt<32>, data : UInt<32>}, flip ready : UInt<1>}
      output chosen : UInt<2>
      reg last-grant17 : UInt<?>
      on-reset last-grant17 := UInt<2>(0)
      node tmp19 = bit-and(in[0].valid, UInt<1>(0))
      node tmp20 = gt(tmp19, last-grant17)
      node tmp21 = bit-and(in[1].valid, UInt<1>(1))
      node tmp22 = gt(tmp21, last-grant17)
      node tmp25 = bit-or(UInt<1>(0), tmp20)
      node tmp26 = bit-not(tmp25)
      node tmp28 = bit-or(UInt<1>(0), tmp20)
      node tmp29 = bit-or(tmp28, tmp22)
      node tmp30 = bit-not(tmp29)
      node tmp32 = bit-or(UInt<1>(0), tmp20)
      node tmp33 = bit-or(tmp32, tmp22)
      node tmp34 = bit-or(tmp33, in[0].valid)
      node tmp35 = bit-not(tmp34)
      node tmp36 = bit-and(UInt<1>(1), UInt<1>(0))
      node tmp37 = gt(tmp36, last-grant17)
      node tmp38 = bit-or(tmp37, tmp30)
      node tmp39 = bit-and(tmp38, out.ready)
      in[0].ready := tmp39
      node tmp40 = bit-and(tmp26, UInt<1>(1))
      node tmp41 = gt(tmp40, last-grant17)
      node tmp42 = bit-or(tmp41, tmp35)
      node tmp43 = bit-and(tmp42, out.ready)
      in[1].ready := tmp43
      accessor a44 = in[chosen]
      out.valid := a44.valid
      accessor a45 = in[chosen]
      out.bits := a45.bits
      wire w46 : UInt<?>
      w46 := UInt<1>(1)
      when in[0].valid :
         w46 := UInt<1>(0)      
      node tmp47 = bit-and(in[1].valid, UInt<1>(1))
      node tmp48 = gt(tmp47, last-grant17)
      wire w49 : UInt<?>
      w49 := w46
      when tmp48 :
         w49 := UInt<1>(1)      
      chosen := w49
      node tmp50 = bit-and(out.ready, out.valid)
      when tmp50 :
         last-grant17 := chosen   
   module SumTop :
      input sreq : {valid : UInt<1>, bits : {v : UInt<32>, len : UInt<32>}, flip ready : UInt<1>}
      output srsp : {valid : UInt<1>, bits : {data : UInt<32>}, flip ready : UInt<1>}
      input mreq : {valid : UInt<1>, bits : {wr? : UInt<1>, tag : UInt<8>, addr : UInt<32>, data : UInt<32>}, flip ready : UInt<1>}
      output mrsp : {valid : UInt<1>, bits : {tag : UInt<8>, data : UInt<32>}, flip ready : UInt<1>}
      inst sum1 of Sum
      inst mem12 of Memory
      inst mreqs16 of ArbiterRR
      mreqs16.in[0] := mreq
      mreqs16.in[1] := sum1.mreq
      mem12.req := mreqs16.out
      sum1.mrsp := mem12.rsp
      mrsp := mem12.rsp
      srsp := sum1.srsp
      sum1.sreq := sreq   

