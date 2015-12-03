 
package midas

import firrtl._
import firrtl.Utils._

/** FAME-1 Transformation
 *
 * This pass takes a lowered-to-ground circuit and performs a FAME-1 (Decoupled) transformation 
 *   to the circuit
 * It does this by creating a simulation module wrapper around the circuit, if we can gate the 
 *   clock, then there need be no modification to the target RTL, if we can't, then the target
 *   RTL will have to be modified (by adding a midasFire input and use this signal to gate 
 *   register enable
 *
 * ALGORITHM
 *  1. Flatten RTL 
 *     a. Create NewTop
 *     b. Instantiate Top in NewTop
 *     c. Iteratively pull all sim tagged instances out of hierarchy to NewTop
 *        i. Move instance declaration from child module to parent module
 *        ii. Create io in child module corresponding to io of instance
 *        iii. Connect original instance io in child to new child io
 *        iv. Connect child io in parent to instance io
 *        v. Repeate until instance is in SimTop 
 *           (if black box, repeat until completely removed from design)
 *     * Post-flattening invariants
 *       - No combinational logic on NewTop
 *       - All and only instances in NewTop are sim tagged
 *       - No black boxes in design
 *  2. Simulation Transformation
 *     //a. Transform Top to SimTop NO THIS STEP IS WRONG AND SHOULDN'T HAPPEN
 *     //   i. Create ready and valid signals for input and output data
 *     //   ii. Create simFire as and of input.valid and output.ready
 *     //   iii. Create simClock as and of clock and simFire
 *     //   iv. Replace clock inputs to sim modules with simClock
 *     b. Iteratively transform each inst in Top (see example firrtl in dram_midas top dir)
 *        i. Create wrapper class
 *        ii. Create input and output (ready, valid) pairs for every 
 *
 * TODO 
 *  - Implement Flatten RTL
 *
 * NOTES 
 *   - How do we only transform the necessary modules? Should there be a MIDAS list of modules 
 *     or something?
 *     * YES, this will be done by requiring the user to instantiate modules that should be split
 *       with something like: val module = new MIDASModule(class... etc.)
 *   - There cannot be nested DecoupledIO or ValidIO
 *   - How do output consumes tie in to MIDAS fire? If all of our outputs are not consumed
 *     in a given cycle, do we block midas$fire on the next cycle? Perhaps there should be 
 *     a register for not having consumed all outputs last cycle
 *   - If our outputs are not consumed we also need to be sure not to consume out inputs,
 *     so the logic for this must depend on the previous cycle being consumed as well
 *   - We also need a way to determine the difference between the MIDAS modules and their
 *     connecting Queues, perhaps they should be MIDAS queues, which then perhaps prints
 *     out a listing of all queues so that they can be properly transformed
 *       * What do these MIDAS queues look like since we're enforcing true decoupled 
 *         interfaces?
 */
object Fame1 {

  //private type PortMap = Map[String, Port]
  //private val PortMap = Map[String, Type]().withDefaultValue(UnknownType)
  private val f1TReady = Field("Ready", Reverse, UIntType(IntWidth(1)))
  private val f1TValid = Field("Valid", Default, UIntType(IntWidth(1)))
  private val f1THostReady = Field("hostReady", Reverse, UIntType(IntWidth(1)))
  private val f1THostValid = Field("hostValid", Default, UIntType(IntWidth(1)))
  private def fame1Transform(t: Type): Type = {
    t match {
      case ClockType => t // Omit clocktype
      case t: BundleType => {
        val names = t.fields.map(_.name)
        if (names.length == 3 && names.contains("ready") && names.contains("valid") && 
            names.contains("bits")) {
          // Decoupled (group valid and bits)
          println("DecoupledIO Detected!")
          t
        //} else if (names.length == 2 && names.contains("valid") && names.contains("bits")) {
        //  //Valid (group valid and bits)
        } else {
          // Default (transform each individually)
          BundleType(t.fields.map(f => Field(f.name, f.dir, fame1Transform(f.tpe))))
        }
      }
      case t: VectorType => 
        VectorType(fame1Transform(t.tpe), t.size)
      case t: Type => 
        BundleType(Seq(f1THostReady, f1THostValid, Field("hostBits", Default, t)))
    }
  }
  private def fame1Transform(p: Port): Port = {
    if( p.name == "reset" ) p // omit reset
    else Port(p.info, p.name, p.dir, fame1Transform(p.tpe))
  }
  private def fame1Transform(m: Module, topName: String): Module = {
    if ( m.name == topName ) m // Skip the top module
    else {
      // Return new wrapper module
      println("fame1Transform called on module " + m.name)
      val ports = m.ports.map(fame1Transform)
      //val portMap = ports.map(p => p.name -> p).toMap
      //val portMap = ports.map(p => p.name -> p)(collection.breakOut): Map[String, Port]
      //println(portMap)
      Module(m.info, m.name, ports, m.stmt)
    }
  }

  private trait SimInstT // TODO Is this name okay?
  private case object UnknownSimInst extends SimInstT
  private case object SimTopIO extends SimInstT
  private case class SimInst(name: String, module: Module, ports: Seq[SimPort]) extends SimInstT
  private case class SimPort(port: Port, endpoint: SimInstT)

  //private def findPortEndpoint(simInsts: Seq[String], port: SimPort): Option[SimInst] = {
  //  SimPort(port, UnknownSimInst)
  //  
  //}

  // Convert firrtl.DefInst to augmented type SimInst
  private def convertInst(c: Circuit, inst: DefInst): SimInst = {
    val moduleName = inst.module match {
      case r: Ref => r.name
      case _ => throw new Exception("Module child of DefInst is not a Ref Exp!") 
    }
    val module = c.modules.find(_.name == moduleName)
    if (module.isEmpty) throw new Exception("No module found with name " + moduleName)
    SimInst(inst.name, module.get, module.get.ports.map(SimPort(_, UnknownSimInst)))
  }

  private def getDefInsts(s: Stmt): Seq[DefInst] = {
    s match {
      case i: DefInst => Seq(i)
      case b: Block => b.stmts.map(getDefInsts).flatten
      case _ => Seq()
    }
  }
  private def getDefInsts(m: Module): Seq[DefInst] = getDefInsts(m.stmt)

  private def getInstRef(inst: DefInst): Ref = {
    inst.module match {
      case ref: Ref => ref
      case _ => throw new Exception("Invalid module expression for DefInst: " + inst.serialize)
    }
  }

  // DefInsts have an Expression for the module, this expression should be a reference this 
  //   reference has a tpe that should be a bundle representing the IO of that module class
  private def getDefInstType(inst: DefInst): BundleType = {
    val ref = getInstRef(inst)
    ref.tpe match {
      case b: BundleType => b
      case _ => throw new Exception("Invalid reference type for DefInst: " + inst.serialize)
    }
  }

  private def getModuleFromDefInst(nameToModule: Map[String, Module], inst: DefInst): Module = {
    val instModule = getInstRef(inst)
    if(!nameToModule.contains(instModule.name))
      throw new Exception(s"Module ${instModule.name} not found in circuit!")
    else 
      nameToModule(instModule.name)
  }

  private def genWrapperModuleName(name: String): String = s"SimWrap_${name}"
  private val readyValidPair = BundleType(Seq(Field("ready", Reverse, UIntType(IntWidth(1))),
                                              Field("valid", Default, UIntType(IntWidth(1)))))
  private def getPortDir(dir: FieldDir): PortDir = {
    dir match {
      case Default => Output
      case Reverse => Input
    }
  }
  // Takes a set of strings and returns equivalent subfield node
  // eg. Seq(io, port, ready) corresponds to io.port.ready
  private def namesToSubfield(names: Seq[String]): Subfield = {
    def rec(names: Seq[String]): Exp = {
      if( names.length == 1 ) Ref(names.head, UnknownType)
      else Subfield(rec(names.tail), names.head, UnknownType)
    }
    rec(names.reverse) match {
      case s: Subfield => s
      case _ => throw new Exception("Subfield requires more than 1 name!")
    }
  }
  private def genAndReduce(args: Seq[Seq[String]]): DoPrimop = {
    if( args.length == 2 ) 
      DoPrimop(And, Seq(namesToSubfield(args.head), namesToSubfield(args.last)), Seq(), UnknownType)
    else 
      DoPrimop(And, Seq(namesToSubfield(args.head), genAndReduce(args.tail)), Seq(), UnknownType)
  }
  private def genWrapperModule(inst: DefInst, connections: Seq[String]): Module = {
    val instIO = getDefInstType(inst)
    // Add ports for each connection
    val simInputPorts = connections.map(s => Port(inst.info, s"simInput_${s}", Input, readyValidPair))
    val simOutputPorts = connections.map(s => Port(inst.info, s"simOutput_${s}", Output, readyValidPair))
    val rtlPorts = instIO.fields.map(f => Port(inst.info, f.name, getPortDir(f.dir), f.tpe))
    val ports = rtlPorts ++ simInputPorts ++ simOutputPorts

    val simFireInputs = simInputPorts.map(p => Seq(p.name, "valid")) ++ simOutputPorts.map(p => Seq(p.name, "ready"))
    val simFire = DefNode(inst.info, "simFire", genAndReduce(simFireInputs))
    val simClock = DefNode(inst.info, "simClock", DoPrimop(And, 
                     Seq(Ref(simFire.name, UnknownType), Ref("clock", UnknownType)), Seq(), UnknownType))
    val inputsReady = simInputPorts.map(p => 
           Connect(inst.info, namesToSubfield(Seq(p.name, "ready")), UIntValue(1, IntWidth(1))))
    val outputsValid = simOutputPorts.map(p =>
           Connect(inst.info, namesToSubfield(Seq(p.name, "valid")), Ref(simFire.name, UnknownType)))
    val instIOConnect = instIO.fields.map{ io => 
        io.tpe match {
          case ClockType => Connect(inst.info, Ref(io.name, io.tpe), Ref(simClock.name, UnknownType))
          case _ =>
            io.dir match {
              case Default => Connect(inst.info, Ref(io.name, io.tpe), namesToSubfield(Seq(inst.name, io.name)))
              case Reverse => Connect(inst.info, namesToSubfield(Seq(inst.name, io.name)), Ref(io.name, io.tpe))
         } } }
    val stmts = Block(Seq(simFire, simClock, inst) ++ inputsReady ++ outputsValid ++ instIOConnect)

    Module(inst.info, s"SimWrap_${inst.name}", ports, stmts)
  }

  def transform(c: Circuit): Circuit = {
    val nameToModule = c.modules.map(m => m.name -> m)(collection.breakOut): Map[String, Module] 
    val top = nameToModule(c.name)

    println("Top Module:")
    println(top.serialize)

    val insts = getDefInsts(top)
    println(s"In top module ${top.name}, we have instances: ")
    insts.foreach(i => println("  " + i.name))

    val connections = Seq("topIO") ++ insts.map(_.name)
    println(connections)

    val wrappers = insts.map { inst =>
      genWrapperModule(inst, connections.filter(_ != inst.name))
    }

    wrappers.foreach { w => println(w.serialize) }

    //val wrappers = insts.map(genWrapperModule(_, connections))
    //wrappers.foreach(println(_.serialize))
      
    c
  }

}
