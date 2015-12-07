
package midas

import firrtl._
import firrtl.Utils._

object Utils {

  // Merges a sequence of maps via the provided function f
  // Taken from: https://groups.google.com/forum/#!topic/scala-user/HaQ4fVRjlnU
  def merge[K, V](maps: Seq[Map[K, V]])(f: (K, V, V) => V): Map[K, V] = {
    maps.foldLeft(Map.empty[K, V]) { case (merged, m) =>
      m.foldLeft(merged) { case (acc, (k, v)) =>
        acc.get(k) match {
          case Some(existing) => acc.updated(k, f(k, existing, v))
          case None => acc.updated(k, v)
        }
      }
    }
  }

  // This doesn't work because of Type Erasure >.<
  //private def getStmts[A <: Stmt](s: Stmt): Seq[A] = {
  //  s match {
  //    case a: A => Seq(a)
  //    case b: Block => b.stmts.map(getStmts[A]).flatten
  //    case _ => Seq()
  //  }
  //}
  //private def getStmts[A <: Stmt](m: Module): Seq[A] = getStmts[A](m.stmt)

  def getDefRegs(s: Stmt): Seq[DefReg] = {
    s match {
      case r: DefReg => Seq(r)
      case b: Block => b.stmts.map(getDefRegs).flatten
      case _ => Seq()
    }
  }
  def getDefRegs(m: Module): Seq[DefReg] = getDefRegs(m.stmt)

  def getDefInsts(s: Stmt): Seq[DefInst] = {
    s match {
      case i: DefInst => Seq(i)
      case b: Block => b.stmts.map(getDefInsts).flatten
      case _ => Seq()
    }
  }
  def getDefInsts(m: Module): Seq[DefInst] = getDefInsts(m.stmt)

  def getDefInstRef(inst: DefInst): Ref = {
    inst.module match {
      case ref: Ref => ref
      case _ => throw new Exception("Invalid module expression for DefInst: " + inst.serialize)
    }
  }

  // DefInsts have an Expression for the module, this expression should be a reference this 
  //   reference has a tpe that should be a bundle representing the IO of that module class
  def getDefInstType(inst: DefInst): BundleType = {
    val ref = getDefInstRef(inst)
    ref.tpe match {
      case b: BundleType => b
      case _ => throw new Exception("Invalid reference type for DefInst: " + inst.serialize)
    }
  }

  def getModuleFromDefInst(nameToModule: Map[String, Module], inst: DefInst): Module = {
    val instModule = getDefInstRef(inst)
    if(!nameToModule.contains(instModule.name))
      throw new Exception(s"Module ${instModule.name} not found in circuit!")
    else 
      nameToModule(instModule.name)
  }

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

  def genPrimopReduce(op: Primop, args: Seq[Exp]): Exp = {
    if( args.length == 0 ) throw new Exception("genPrimopReduce called on empty sequence!")
    else if( args.length == 1 ) args.head
    else if( args.length == 2 ) DoPrimop(op, Seq(args.head, args.last), Seq(), UnknownType)
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

  // Recursively iterates through firrtl.Type returning sequence of names to address signals
  //  * Intended for use with recursive bundle types
  def enumerateMembers(tpe: Type): Seq[Seq[Any]] = {
    def rec(tpe: Type, path: Seq[Any], members: Seq[Seq[Any]]): Seq[Seq[Any]] = {
      tpe match {
        case b: BundleType => (b.fields map ( f => rec(f.tpe, path :+ f.name, members) )).flatten
        case v: VectorType => (Seq.tabulate(v.size.toInt) ( i => rec(v.tpe, path :+ i, members) )).flatten
        case _ => members :+ path
      }
    }
    rec(tpe, Seq[Any](), Seq[Seq[Any]]())
  }

  // Queue
  // TODO
  //  - Insert null tokens upon hostReset (or should this be elsewhere?)
  def buildSimQueue(name: String, tpe: Type): Module = {
    val scopeSpaces = " " * 4 // Spaces before lines in module scope, for default assignments
    val templatedQueue =
//      """
//    circuit `NAME:
//      module `NAME : 
//        input hostClock : Clock
//        input hostReset : UInt<1>
//        output io : {flip enq : {flip hostReady : UInt<1>, hostValid : UInt<1>, hostBits : `TYPE}, deq : {flip hostReady : UInt<1>, hostValid : UInt<1>, hostBits : `TYPE}, count : UInt<3>}
//        
//        io.count := UInt<1>("h00")
//        `DEFAULT_ASSIGN
//        io.deq.hostValid := UInt<1>("h00")
//        io.enq.hostReady := UInt<1>("h00")
//        cmem ram : `TYPE[4], hostClock
//        reg T_80 : UInt<2>, hostClock, hostReset
//        onreset T_80 := UInt<2>("h00")
//        reg T_82 : UInt<2>, hostClock, hostReset
//        onreset T_82 := UInt<2>("h00")
//        reg maybe_full : UInt<1>, hostClock, hostReset
//        onreset maybe_full := UInt<1>("h00")
//        node ptr_match = eq(T_80, T_82)
//        node T_87 = eq(maybe_full, UInt<1>("h00"))
//        node empty = and(ptr_match, T_87)
//        node full = and(ptr_match, maybe_full)
//        node maybe_flow = and(UInt<1>("h00"), empty)
//        node do_flow = and(maybe_flow, io.deq.hostReady)
//        node T_93 = and(io.enq.hostReady, io.enq.hostValid)
//        node T_95 = eq(do_flow, UInt<1>("h00"))
//        node do_enq = and(T_93, T_95)
//        node T_97 = and(io.deq.hostReady, io.deq.hostValid)
//        node T_99 = eq(do_flow, UInt<1>("h00"))
//        node do_deq = and(T_97, T_99)
//        when do_enq :
//          infer accessor T_101 = ram[T_80]
//          T_101 <> io.enq.hostBits
//          node T_109 = eq(T_80, UInt<2>("h03"))
//          node T_111 = and(UInt<1>("h00"), T_109)
//          node T_114 = addw(T_80, UInt<1>("h01"))
//          node T_115 = mux(T_111, UInt<1>("h00"), T_114)
//          T_80 := T_115
//          skip
//        when do_deq :
//          node T_117 = eq(T_82, UInt<2>("h03"))
//          node T_119 = and(UInt<1>("h00"), T_117)
//          node T_122 = addw(T_82, UInt<1>("h01"))
//          node T_123 = mux(T_119, UInt<1>("h00"), T_122)
//          T_82 := T_123
//          skip
//        node T_124 = neq(do_enq, do_deq)
//        when T_124 :
//          maybe_full := do_enq
//          skip
//        node T_126 = eq(empty, UInt<1>("h00"))
//        node T_128 = and(UInt<1>("h00"), io.enq.hostValid)
//        node T_129 = or(T_126, T_128)
//        io.deq.hostValid := T_129
//        node T_131 = eq(full, UInt<1>("h00"))
//        node T_133 = and(UInt<1>("h00"), io.deq.hostReady)
//        node T_134 = or(T_131, T_133)
//        io.enq.hostReady := T_134
//        infer accessor T_135 = ram[T_82]
//        wire T_149 : `TYPE
//        T_149 <> T_135
//        when maybe_flow :
//          T_149 <> io.enq.hostBits
//          skip
//        io.deq.hostBits <> T_149
//        node ptr_diff = subw(T_80, T_82)
//        node T_157 = and(maybe_full, ptr_match)
//        node T_158 = cat(T_157, ptr_diff)
//        io.count := T_158
//      """
      """
circuit `NAME:
  module `NAME : 
    input hostClock : Clock
    input hostReset : UInt<1>
    output io : {flip enq : {flip hostReady : UInt<1>, hostValid : UInt<1>, hostBits : `TYPE}, deq : {flip hostReady : UInt<1>, hostValid : UInt<1>, hostBits : `TYPE}, count : UInt<3>}
    
    io.count := UInt<1>("h00")
    `DEFAULT_ASSIGN
    io.deq.hostValid := UInt<1>("h00")
    io.enq.hostReady := UInt<1>("h00")
    cmem ram : `TYPE[4], hostClock
    reg T_404 : UInt<2>, hostClock, hostReset
    onreset T_404 := UInt<2>("h00")
    reg T_406 : UInt<2>, hostClock, hostReset
    onreset T_406 := UInt<2>("h00")
    reg maybe_full : UInt<1>, hostClock, hostReset
    onreset maybe_full := UInt<1>("h00")
    reg add_token_on_reset : UInt<1>, hostClock, hostReset
    onreset add_token_on_reset := UInt<1>("h01")
    add_token_on_reset := UInt<1>("h00")
    node ptr_match = eq(T_404, T_406)
    node T_414 = eq(maybe_full, UInt<1>("h00"))
    node empty = and(ptr_match, T_414)
    node full = and(ptr_match, maybe_full)
    node maybe_flow = and(UInt<1>("h00"), empty)
    node do_flow = and(maybe_flow, io.deq.hostReady)
    node T_420 = and(io.enq.hostReady, io.enq.hostValid)
    node T_422 = eq(do_flow, UInt<1>("h00"))
    node do_enq = and(T_420, T_422)
    node T_424 = and(io.deq.hostReady, io.deq.hostValid)
    node T_426 = eq(do_flow, UInt<1>("h00"))
    node do_deq = and(T_424, T_426)
    node T_428 = or(do_enq, add_token_on_reset)
    when T_428 :
      infer accessor T_443 = ram[T_404]
      T_443 := io.enq.hostBits
      node T_473 = eq(T_404, UInt<2>("h03"))
      node T_475 = and(UInt<1>("h00"), T_473)
      node T_478 = addw(T_404, UInt<1>("h01"))
      node T_479 = mux(T_475, UInt<1>("h00"), T_478)
      T_404 := T_479
      skip
    when do_deq :
      node T_481 = eq(T_406, UInt<2>("h03"))
      node T_483 = and(UInt<1>("h00"), T_481)
      node T_486 = addw(T_406, UInt<1>("h01"))
      node T_487 = mux(T_483, UInt<1>("h00"), T_486)
      T_406 := T_487
      skip
    node T_488 = neq(do_enq, do_deq)
    when T_488 :
      maybe_full := do_enq
      skip
    node T_490 = eq(empty, UInt<1>("h00"))
    node T_492 = and(UInt<1>("h00"), io.enq.hostValid)
    node T_493 = or(T_490, T_492)
    io.deq.hostValid := T_493
    node T_495 = eq(full, UInt<1>("h00"))
    node T_497 = and(UInt<1>("h00"), io.deq.hostReady)
    node T_498 = or(T_495, T_497)
    io.enq.hostReady := T_498
    infer accessor T_513 = ram[T_406]
    wire T_599 : `TYPE
    T_599 := T_513
    when maybe_flow :
      T_599 := io.enq.hostBits
      skip
    io.deq.hostBits := T_599
    node ptr_diff = subw(T_404, T_406)
    node T_629 = and(maybe_full, ptr_match)
    node T_630 = cat(T_629, ptr_diff)
    io.count := T_630
      """
    // Generate initial values
    val signals = enumerateMembers(tpe) map ( Seq("io", "deq", "hostBits") ++ _ )
    val defaultAssign = signals map { sig =>
      scopeSpaces + Connect(NoInfo, buildExp(sig), UIntValue(0, UnknownWidth)).serialize
    }

    val concreteQueue = templatedQueue.replaceAllLiterally("`NAME", name).
                                       replaceAllLiterally("`TYPE", tpe.serialize).
                                       replaceAllLiterally(scopeSpaces+"`DEFAULT_ASSIGN", defaultAssign.mkString("\n"))

    val ast = firrtl.Parser.parse(concreteQueue.split("\n"))
    ast.modules.head
  }

}
