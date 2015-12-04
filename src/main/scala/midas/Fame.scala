 
package midas

import Utils._
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
 *     a. Iteratively transform each inst in Top (see example firrtl in dram_midas top dir)
 *        i.   Create wrapper class
 *        ii.  Create input and output (ready, valid) pairs for every other sim module this module connects to
 *             * Note that TopIO counts as a "sim module"
 *        iii. Create simFire as AND of inputs.valid and outputs.ready
 *        iv.  Create [target] simClock as AND of simFire and [host] clock
 *        v.   Connect target IO to wrapper IO, except connect target clock to simClock
 *
 * TODO 
 *  - Implement Flatten RTL
 *  - Check that circuit is in LowFIRRTL?
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

  private def getDefInsts(s: Stmt): Seq[DefInst] = {
    s match {
      case i: DefInst => Seq(i)
      case b: Block => b.stmts.map(getDefInsts).flatten
      case _ => Seq()
    }
  }
  private def getDefInsts(m: Module): Seq[DefInst] = getDefInsts(m.stmt)

  private def getDefInstRef(inst: DefInst): Ref = {
    inst.module match {
      case ref: Ref => ref
      case _ => throw new Exception("Invalid module expression for DefInst: " + inst.serialize)
    }
  }

  // DefInsts have an Expression for the module, this expression should be a reference this 
  //   reference has a tpe that should be a bundle representing the IO of that module class
  private def getDefInstType(inst: DefInst): BundleType = {
    val ref = getDefInstRef(inst)
    ref.tpe match {
      case b: BundleType => b
      case _ => throw new Exception("Invalid reference type for DefInst: " + inst.serialize)
    }
  }

  private def getModuleFromDefInst(nameToModule: Map[String, Module], inst: DefInst): Module = {
    val instModule = getDefInstRef(inst)
    if(!nameToModule.contains(instModule.name))
      throw new Exception(s"Module ${instModule.name} not found in circuit!")
    else 
      nameToModule(instModule.name)
  }

  // ***** findPortConn *****
  // This takes lowFIRRTL top module that follows invariants described above and returns a connection Map
  //   of instanceName -> (instanctPorts -> portEndpoint)
  // It honestly feels kind of brittle given it assumes there will be no intermediate nodes or anything in 
  //  the way of direct connections between IO of module instances
  private type PortMap = Map[String, String]
  private val  PortMap = Map[String, String]()
  private type ConnMap = Map[String, PortMap]
  private val  ConnMap = Map[String, PortMap]()
  private def processConnectExp(exp: Exp): (String, String) = {
    val unsupportedExp = new Exception("Unsupported Exp for finding port connections: " + exp)
    exp match {
      case ref: Ref => ("topIO", ref.name)
      case sub: Subfield => 
        sub.exp match {
          case ref: Ref => (ref.name, sub.name)
          case _ => throw unsupportedExp
        }
      case exp: Exp => throw unsupportedExp
    }
  }
  private def processConnect(conn: Connect): ConnMap = {
    val lhs = processConnectExp(conn.lhs)
    val rhs = processConnectExp(conn.rhs)
    Map(lhs._1 -> Map(lhs._2 -> rhs._1), rhs._1 -> Map(rhs._2 -> lhs._1)).withDefaultValue(PortMap)
  }
  private def findPortConn(connMap: ConnMap, stmts: Seq[Stmt]): ConnMap = {
    if (stmts.isEmpty) connMap
    else {
      stmts.head match {
        case conn: Connect => {
          val newConnMap = processConnect(conn)
          findPortConn(connMap.map{case (k,v) => k -> (v ++ newConnMap(k)) }, stmts.tail)
        }
        case _ => findPortConn(connMap, stmts.tail)
      }
    }
  }
  private def findPortConn(top: Module, insts: Seq[DefInst]): ConnMap = {
    val initConnMap = insts.map( _.name -> PortMap ).toMap ++ Map("topIO" -> PortMap)
    val topStmts = top.stmt match {
      case b: Block => b.stmts
      case s: Stmt => Seq(s) // This honestly shouldn't happen but let's be safe
    }
    findPortConn(initConnMap, topStmts)
  }

  // ***** genWrapperModule *****
  // Generates FAME-1 Decoupled wrappers for simulation module instances
  private val readyField = Field("ready", Reverse, UIntType(IntWidth(1)))
  private val validField = Field("valid", Default, UIntType(IntWidth(1)))

  private def genWrapperModule(inst: DefInst, portMap: PortMap): Module = {
    val instIO = getDefInstType(inst)
    val nameToField = instIO.fields.map(f => f.name -> f).toMap

    val connections = portMap.map(_._2).toSeq.distinct // modules we connect to
    // Build simPort for each connecting module
    val connPorts = connections.map{ c =>
      // Get ports that connect to this particular module as fields
      val fields = portMap.filter(_._2 == c).keySet.toSeq.sorted.map(nameToField(_))
      val noClock = fields.filter(_.tpe != ClockType) // Remove clock
      val inputSet  = noClock.filter(_.dir == Reverse).map(f => Field(f.name, Default, f.tpe))
      val outputSet = noClock.filter(_.dir == Default)
      Port(inst.info, c + "Port", Output, BundleType(Seq( 
        Field("input", Reverse, BundleType(Seq(readyField, validField) ++ inputSet)),
        Field("output", Default, BundleType(Seq(readyField, validField) ++ outputSet))
      )))
    }
    val ports = connPorts ++ instIO.fields.filter(_.tpe == ClockType).map(_.toPort) // Add clock back

    // simFire is signal to indicate when a simulation module can execute, this is indicated by all of its inputs
    //   being valid and all of its outputs being ready
    val simFireInputs = connPorts.map { port => 
      getFields(port).map { field => 
        field.dir match {
          case Reverse => buildExp(Seq(port.name, field.name, validField.name)) 
          case Default => buildExp(Seq(port.name, field.name, readyField.name))
        }
      }
    }.flatten
    val simFire = DefNode(inst.info, "simFire", genPrimopReduce(And, simFireInputs))
    // simClock is the simple AND of simFire and the real clock so that the rtl module only executes when data is 
    //   available and outputs are ready
    val simClock = DefNode(inst.info, "simClock", DoPrimop(And, 
                     Seq(Ref(simFire.name, UnknownType), Ref("clock", UnknownType)), Seq(), UnknownType))
    // As a simple RTL module, we're always ready
    val inputsReady = connPorts.map { port => 
      getFields(port).filter(_.dir == Reverse).map { field =>
        Connect(inst.info, buildExp(Seq(port.name, field.name, readyField.name)), UIntValue(1, IntWidth(1)))
      }
    }.flatten
    // Outputs are valid on cycles where we fire
    val outputsValid = connPorts.map { port => 
      getFields(port).filter(_.dir == Default).map { field =>
        Connect(inst.info, buildExp(Seq(port.name, field.name, validField.name)), Ref(simFire.name, UnknownType))
      }
    }.flatten
    //val outputsValid = simOutputPorts.map(p =>
    //       Connect(inst.info, buildExp(Seq(p.name, "valid")), Ref(simFire.name, UnknownType)))
    //// Connect up all of the IO of the RTL module to sim module IO, except clock which should be connected
    ////   to simClock
    //val instIOConnect = instIO.fields.map{ io => 
    //    io.tpe match {
    //      case ClockType => Connect(inst.info, Ref(io.name, io.tpe), Ref(simClock.name, UnknownType))
    //      case _ =>
    //        io.dir match {
    //          case Default => Connect(inst.info, Ref(io.name, io.tpe), buildExp(Seq(inst.name, io.name)))
    //          case Reverse => Connect(inst.info, buildExp(Seq(inst.name, io.name)), Ref(io.name, io.tpe))
    //     } } }
    //val stmts = Block(Seq(simFire, simClock, inst) ++ inputsReady ++ outputsValid ++ instIOConnect)
    val stmts = Block(Seq(simFire, simClock) ++ inputsReady ++ outputsValid)

    Module(inst.info, s"SimWrap_${inst.name}", ports, stmts)
  }

  // ***** transform *****
  // Perform FAME-1 Transformation for MIDAS
  def transform(c: Circuit): Circuit = {
    // We should do a check low firrtl
    val nameToModule = c.modules.map(m => m.name -> m)(collection.breakOut): Map[String, Module] 
    val top = nameToModule(c.name)

    println("Top Module:")
    println(top.serialize)

    val insts = getDefInsts(top)

    val portConn = findPortConn(top, insts)

    val wrappers = insts.map { inst =>
      genWrapperModule(inst, portConn(inst.name))
    }

    wrappers.foreach { w => println(w.serialize) }

    c
  }

}
