
package midas

import firrtl._

object Utils {

  // Takes a set of strings or ints and returns equivalent expression node
  //   Strings correspond to subfields/references, ints correspond to indexes
  // eg. Seq(io, port, ready)    => io.port.ready
  //     Seq(io, port, 5, valid) => io.port[5].valid
  //     Seq(3)                  => UInt("h3")
  def buildExp(names: Seq[Any]): Exp = {
    def rec(names: Seq[Any]): Exp = {
      names.head match {
        // Useful for adding on indexes or subfields
        case head: Exp => head 
        // Int -> UInt/SInt/Index
        case head: Int => 
          if( names.tail.isEmpty ) // Is the UInt/SInt inference good enough?
            if( head > 0 ) UIntValue(head, UnknownWidth) else SIntValue(head, UnknownWidth)
          else Index(rec(names.tail), head, UnknownType)
        // String -> Ref/Subfield
        case head: String => 
          if( names.tail.isEmpty ) Ref(head, UnknownType)
          else Subfield(rec(names.tail), head, UnknownType)
        case _ => throw new Exception("Invalid argument type to buildExp! " + names)
      }
    }
    rec(names.reverse) // Let user specify in more natural format
  }
  def buildExp(name: Any): Exp = buildExp(Seq(name))

  def genPrimopReduce(op: Primop, args: Seq[Exp]): DoPrimop = {
    if( args.length == 2 ) DoPrimop(op, Seq(args.head, args.last), Seq(), UnknownType)
    else DoPrimop(op, Seq(args.head, genPrimopReduce(op, args.tail)), Seq(), UnknownType)
  }

  // For a port that is known to be of type BundleType, return the fields of that bundle
  def getFields(port: Port): Seq[Field] = {
    port.tpe match {
      case b: BundleType => b.fields
      case _ => throw new Exception("getFields called on invalid port " + port)
    }
  }

  // Queue
  /*
   module Queue :
    input clock : Clock
    input reset : UInt<1>
    output io : {flip enq : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<32>}, deq : {flip ready : UInt<1>, valid : UInt<1>, bits : UInt<32>}, count : UInt<3>}

    io.count := UInt<1>("h00")
    io.deq.bits := UInt<1>("h00")
    io.deq.valid := UInt<1>("h00")
    io.enq.ready := UInt<1>("h00")
    cmem ram : UInt<32>[4], clock
    reg T_26 : UInt<2>, clock, reset
    onreset T_26 := UInt<2>("h00")
    reg T_28 : UInt<2>, clock, reset
    onreset T_28 := UInt<2>("h00")
    reg maybe_full : UInt<1>, clock, reset
    onreset maybe_full := UInt<1>("h00")
    node ptr_match = eq(T_26, T_28)
    node T_33 = eq(maybe_full, UInt<1>("h00"))
    node empty = and(ptr_match, T_33)
    node full = and(ptr_match, maybe_full)
    node maybe_flow = and(UInt<1>("h00"), empty)
    node do_flow = and(maybe_flow, io.deq.ready)
    node T_39 = and(io.enq.ready, io.enq.valid)
    node T_41 = eq(do_flow, UInt<1>("h00"))
    node do_enq = and(T_39, T_41)
    node T_43 = and(io.deq.ready, io.deq.valid)
    node T_45 = eq(do_flow, UInt<1>("h00"))
    node do_deq = and(T_43, T_45)
    when do_enq :
      infer accessor T_47 = ram[T_26]
      T_47 := io.enq.bits
      node T_49 = eq(T_26, UInt<2>("h03"))                                                                                                                                                                                                    
      node T_51 = and(UInt<1>("h00"), T_49)
      node T_54 = addw(T_26, UInt<1>("h01"))
      node T_55 = mux(T_51, UInt<1>("h00"), T_54)
      T_26 := T_55
      skip
    when do_deq :
      node T_57 = eq(T_28, UInt<2>("h03"))
      node T_59 = and(UInt<1>("h00"), T_57)
      node T_62 = addw(T_28, UInt<1>("h01"))
      node T_63 = mux(T_59, UInt<1>("h00"), T_62)
      T_28 := T_63
      skip
    node T_64 = neq(do_enq, do_deq)
    when T_64 :
      maybe_full := do_enq
      skip
    node T_66 = eq(empty, UInt<1>("h00"))
    node T_68 = and(UInt<1>("h00"), io.enq.valid)
    node T_69 = or(T_66, T_68)
    io.deq.valid := T_69
    node T_71 = eq(full, UInt<1>("h00"))
    node T_73 = and(UInt<1>("h00"), io.deq.ready)
    node T_74 = or(T_71, T_73)
    io.enq.ready := T_74
    infer accessor T_75 = ram[T_28]
    node T_76 = mux(maybe_flow, io.enq.bits, T_75)
    io.deq.bits := T_76
    node ptr_diff = subw(T_26, T_28)
    node T_78 = and(maybe_full, ptr_match)
    node T_79 = cat(T_78, ptr_diff)
    io.count := T_79
  */ 

}
