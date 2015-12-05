
package midas

import firrtl._
import firrtl.Utils._

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
          if( names.tail.isEmpty ) Ref("head", UnknownType)
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

  // Checks if a firrtl.Port matches the MIDAS SimPort pattern
  // This currently just checks that the port is of type bundle with ONLY the members
  //   hostIn and/or hostOut with correct directions
  def isSimPort(port: Port): Boolean = {
    //println("isSimPort called on port " + port.serialize)
    port.tpe match {
      case b: BundleType => {
        b.fields map { field =>
          if( field.name == "hostIn" ) field.dir == Reverse
          else if( field.name == "hostOut" ) field.dir == Default
          else false
        } reduce ( _ & _ )
      }
      case _ => false
    }
  }

  def splitSimPort(port: Port): Seq[Field] = {
    try {
      val b = port.tpe.asInstanceOf[BundleType]
      Seq(b.fields.find(_.name == "hostIn"), b.fields.find(_.name == "hostOut")).flatten
    } catch {
      case e: Exception => throw new Exception("Invalid SimPort " + port.serialize)
    }
  }

  // From simulation host decoupled, return hostbits field 
  def getHostBits(field: Field): Field = {
    try {
      val b = field.tpe.asInstanceOf[BundleType]
      b.fields.find(_.name == "hostBits").get
    } catch {
      case e: Exception => throw new Exception("Invalid SimField " + field.serialize)
    }
  }

  // For a port that is known to be of type BundleType, return the fields of that bundle
  def getFields(port: Port): Seq[Field] = {
    port.tpe match {
      case b: BundleType => b.fields
      case _ => throw new Exception("getFields called on invalid port " + port)
    }
  }

  // Queue
  def buildSimQueue(name: String, tpe: Type): Module = {
    val templatedQueue = 
      """
      module `NAME : 
        input clock : Clock
        input reset : UInt<1>
        output io : {flip enq : {flip ready : UInt<1>, valid : UInt<1>, bits : `TYPE}, deq : {flip ready : UInt<1>, valid : UInt<1>, bits : `TYPE}, count : UInt<3>}
        
        io.count := UInt<1>("h00")
        io.deq.bits.surprise.no := UInt<1>("h00")
        io.deq.bits.surprise.yes := UInt<1>("h00")
        io.deq.bits.store := UInt<1>("h00")
        io.deq.bits.data := UInt<1>("h00")
        io.deq.bits.addr := UInt<1>("h00")
        io.deq.valid := UInt<1>("h00")
        io.enq.ready := UInt<1>("h00")
        cmem ram : `TYPE[4], clock
        reg T_80 : UInt<2>, clock, reset
        onreset T_80 := UInt<2>("h00")
        reg T_82 : UInt<2>, clock, reset
        onreset T_82 := UInt<2>("h00")
        reg maybe_full : UInt<1>, clock, reset
        onreset maybe_full := UInt<1>("h00")
        node ptr_match = eq(T_80, T_82)
        node T_87 = eq(maybe_full, UInt<1>("h00"))
        node empty = and(ptr_match, T_87)
        node full = and(ptr_match, maybe_full)
        node maybe_flow = and(UInt<1>("h00"), empty)
        node do_flow = and(maybe_flow, io.deq.ready)
        node T_93 = and(io.enq.ready, io.enq.valid)
        node T_95 = eq(do_flow, UInt<1>("h00"))
        node do_enq = and(T_93, T_95)
        node T_97 = and(io.deq.ready, io.deq.valid)
        node T_99 = eq(do_flow, UInt<1>("h00"))
        node do_deq = and(T_97, T_99)
        when do_enq :
          infer accessor T_101 = ram[T_80]
          T_101 <> io.enq.bits
          node T_109 = eq(T_80, UInt<2>("h03"))
          node T_111 = and(UInt<1>("h00"), T_109)
          node T_114 = addw(T_80, UInt<1>("h01"))
          node T_115 = mux(T_111, UInt<1>("h00"), T_114)
          T_80 := T_115
          skip
        when do_deq :
          node T_117 = eq(T_82, UInt<2>("h03"))
          node T_119 = and(UInt<1>("h00"), T_117)
          node T_122 = addw(T_82, UInt<1>("h01"))
          node T_123 = mux(T_119, UInt<1>("h00"), T_122)
          T_82 := T_123
          skip
        node T_124 = neq(do_enq, do_deq)
        when T_124 :
          maybe_full := do_enq
          skip
        node T_126 = eq(empty, UInt<1>("h00"))
        node T_128 = and(UInt<1>("h00"), io.enq.valid)
        node T_129 = or(T_126, T_128)
        io.deq.valid := T_129
        node T_131 = eq(full, UInt<1>("h00"))
        node T_133 = and(UInt<1>("h00"), io.deq.ready)
        node T_134 = or(T_131, T_133)
        io.enq.ready := T_134
        infer accessor T_135 = ram[T_82]
        wire T_149 : `TYPE
        T_149 <> T_135
        when maybe_flow :
          T_149 <> io.enq.bits
          skip
        io.deq.bits <> T_149
        node ptr_diff = subw(T_80, T_82)
        node T_157 = and(maybe_full, ptr_match)
        node T_158 = cat(T_157, ptr_diff)
        io.count := T_158
      """
    //def buildQueue(name: String, tpe: Type): Module = {
    val concreteQueue = templatedQueue.replaceAllLiterally("`NAME", name).
                                       replaceAllLiterally("`TYPE", tpe.serialize)
    // Generate initial values
    //val bitsField = Field("bits", Default, tpe)
    println(concreteQueue.stripMargin)
    firrtl.Parser.parseModule(concreteQueue)
  }

}
