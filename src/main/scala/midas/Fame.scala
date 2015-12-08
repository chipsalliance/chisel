 
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
 *     a. Perform Decoupled Transformation on every RTL Module
 *        i.   Add targetFire signal input
 *        ii.  Find all DefInst nodes and propogate targetFire to the instances
 *        iii. Find all registers and add when statement connecting regIn to regOut when !targetFire
 *     b. Iteratively transform each inst in Top (see example firrtl in dram_midas top dir)
 *        i.   Create wrapper class
 *        ii.  Create input and output (ready, valid) pairs for every other sim module this module connects to
 *             * Note that TopIO counts as a "sim module"
 *        iii. Create targetFire as AND of inputs.valid and outputs.ready
 *        iv.  Connect targetFire to targetFire input of target rtl inst
 *        v.   Connect target IO to wrapper IO, except connect target clock to simClock
 *
 * TODO 
 *  - Is it okay to have ready signals for input queues depend on valid signals for those queues? This is generally bad
 *  - Change sequential memory read enable to work with targetFire
 *  - Implement Flatten RTL
 *  - Refactor important strings/naming to API (eg. "topIO" needs to be a constant defined somewhere or something)
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

  // Constants, common nodes, and common types used throughout
  private type PortMap = Map[String, Seq[String]]
  private val  PortMap = Map[String, Seq[String]]()
  private type ConnMap = Map[String, PortMap]
  private val  ConnMap = Map[String, PortMap]()
  private type SimQMap = Map[(String, String), Module]
  private val  SimQMap = Map[(String, String), Module]()

  private val hostReady = Field("hostReady", Reverse, UIntType(IntWidth(1)))
  private val hostValid = Field("hostValid", Default, UIntType(IntWidth(1)))
  private val hostClock = Port(NoInfo, "hostClock", Input, ClockType)
  private val hostReset = Port(NoInfo, "hostReset", Input, UIntType(IntWidth(1)))
  private val targetFire = Port(NoInfo, "targetFire", Input, UIntType(IntWidth(1)))

  private def wrapName(name: String): String = s"SimWrap_${name}"
  private def unwrapName(name: String): String = name.stripPrefix("SimWrap_")
  private def queueName(src: String, dst: String): String = s"SimQueue_${src}_${dst}"
  private def instName(name: String): String = s"inst_${name}"
  private def unInstName(name: String): String = name.stripPrefix("inst_")

  private def genHostDecoupled(fields: Seq[Field]): BundleType = {
    BundleType(Seq(hostReady, hostValid) :+ Field("hostBits", Default, BundleType(fields)))
  }

  // ********** findPortConn **********
  // This takes lowFIRRTL top module that follows invariants described above and returns a connection Map
  //   of instanceName -> (instanctPorts -> portEndpoint)
  // It honestly feels kind of brittle given it assumes there will be no intermediate nodes or anything in 
  //  the way of direct connections between IO of module instances
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
    Map(lhs._1 -> Map(lhs._2 -> Seq(rhs._1)), rhs._1 -> Map(rhs._2 -> Seq(lhs._1))).withDefaultValue(PortMap)
  }
  private def findPortConn(connMap: ConnMap, stmts: Seq[Stmt]): ConnMap = {
    if (stmts.isEmpty) connMap
    else {
      stmts.head match {
        case conn: Connect => {
          val newConnMap = processConnect(conn)
          findPortConn((connMap map { case (k,v) => 
            k -> merge(Seq(v, newConnMap(k))) { (_, v1, v2) => v1 ++ v2 }}), stmts.tail )
        }
        case _ => findPortConn(connMap, stmts.tail)
      }
    }
  }
  private def findPortConn(top: Module, insts: Seq[DefInst]): ConnMap = {
    val initConnMap = (insts map ( _.name -> PortMap )).toMap ++ Map("topIO" -> PortMap)
    val topStmts = top.stmt match {
      case b: Block => b.stmts
      case s: Stmt => Seq(s) // This honestly shouldn't happen but let's be safe
    }
    findPortConn(initConnMap, topStmts)
  }

  // Removes clocks from a portmap
  private def scrubClocks(ports: Seq[Port], portMap: PortMap): PortMap = {
    val clocks = ports filter (_.tpe == ClockType) map (_.name)
    portMap filter { case (portName, _) => !clocks.contains(portName) }
  }

  // ********** transformRTL **********
  // Takes an RTL module and give it targetFire input, propogates targetFire to all child instances,
  //   puts targetFire on regEnable for all registers
  // TODO
  //  - Add smem support
  private def transformRTL(m: Module): Module = {
    val ports = m.ports :+ targetFire
    val instProp = getDefInsts(m) map { inst =>
      Connect(NoInfo, buildExp(Seq(inst.name, targetFire.name)), buildExp(targetFire.name))
    }
    val regEn  = getDefRegs(m) map { reg =>
      When(NoInfo, DoPrimop(Not, Seq(buildExp(targetFire.name)), Seq(), UnknownType),
                   Connect(NoInfo, buildExp(reg.name), buildExp(reg.name)), EmptyStmt)
    }
    Module(m.info, m.name, ports, Block(m.stmt +: (instProp ++ regEn)))
  }

  // ********** genWrapperModule **********
  // Generates FAME-1 Decoupled wrappers for simulation module instances
  private def genWrapperModule(inst: DefInst, portMap: PortMap): Module = {

    val instIO = getDefInstType(inst)
    val nameToField = (instIO.fields map (f => f.name -> f)).toMap

    val connections = (portMap map (_._2)).toSeq.flatten.distinct // modules this inst connects to
    // Build simPort for each connecting module
    // TODO This whole chunk really ought to be rewritten or made a function
    val connPorts = connections map { c =>
      // Get ports that connect to this particular module as fields
      val fields = (portMap filter (_._2.contains(c))).keySet.toSeq.sorted map (nameToField(_))
      val noClock = fields filter (_.tpe != ClockType) // Remove clock
      val inputSet  = noClock filter (_.dir == Reverse) map (f => Field(f.name, Default, f.tpe))
      val outputSet = noClock filter (_.dir == Default)
      Port(inst.info, c, Output, BundleType(
        (if (inputSet.isEmpty) Seq()
        else Seq(Field("hostIn", Reverse, genHostDecoupled(inputSet)))
        ) ++
        (if (outputSet.isEmpty) Seq()
        else Seq(Field("hostOut", Default, genHostDecoupled(outputSet)))
        )
      ))
    }
    val ports = hostClock +: hostReset +: connPorts // Add host and host reset

    // targetFire is signal to indicate when a simulation module can execute, this is indicated by all of its inputs
    //   being valid and all of its outputs being ready
    val targetFireInputs = (connPorts map { port => 
      getFields(port) map { field => 
        field.dir match {
          case Reverse => buildExp(Seq(port.name, field.name, hostValid.name)) 
          case Default => buildExp(Seq(port.name, field.name, hostReady.name))
        }
      }
    }).flatten

    val defTargetFire = DefNode(inst.info, targetFire.name, genPrimopReduce(And, targetFireInputs))
    val connectTargetFire = Connect(NoInfo, buildExp(Seq(inst.name, targetFire.name)), buildExp(targetFire.name))

    // Only consume tokens when the module fires
    // TODO is it bad to have the input readys depend on the input valid signals?
    val inputsReady = (connPorts map { port => 
      getFields(port) filter (_.dir == Reverse) map { field => // filter to only take inputs
        Connect(inst.info, buildExp(Seq(port.name, field.name, hostReady.name)), buildExp(targetFire.name))
      }
    }).flatten

    // Outputs are valid on cycles where we fire
    val outputsValid = (connPorts map { port => 
      getFields(port) filter (_.dir == Default) map { field => // filter to only take outputs
        Connect(inst.info, buildExp(Seq(port.name, field.name, hostValid.name)), buildExp(targetFire.name))
      }
    }).flatten

    // Connect up all of the IO of the RTL module to sim module IO, except clock which should be connected
    // This currently assumes naming things that are also done above when generating connPorts
    val connectedInstIOFields = instIO.fields filter(field => portMap.contains(field.name)) // skip unconnected IO
    val instIOConnect = (connectedInstIOFields map { field =>
      field.tpe match {             
        case ClockType => Seq(Connect(inst.info, buildExp(Seq(inst.name, field.name)), 
                                                 Ref(hostClock.name, ClockType)))
        case _ => field.dir match {
          case Default => portMap(field.name) map { endpoint =>
              Connect(inst.info, buildExp(Seq(endpoint, "hostOut", "hostBits", field.name)), 
                                 buildExp(Seq(inst.name, field.name))) 
          }
          case Reverse => { 
              if (portMap(field.name).length > 1) 
                throw new Exception("It is illegal to have more than 1 connection to a single input" + field)
              Seq(Connect(inst.info, buildExp(Seq(inst.name, field.name)),
                                     buildExp(Seq(portMap(field.name).head, "hostIn", "hostBits", field.name))))
          }
        }
      }
    }).flatten
    val stmts = Block(Seq(defTargetFire) ++ inputsReady ++ outputsValid ++ Seq(inst) ++ 
                      Seq(connectTargetFire) ++ instIOConnect)

    Module(inst.info, wrapName(inst.name), ports, stmts)
  }

  // ********** generateSimQueues **********
  // Takes Seq of SimWrapper modules
  // Returns Map of (src, dest) -> SimQueue
  // To prevent duplicates, instead of creating a map with (src, dest) as the key, we could instead
  //   only one direction of the queue for each simport of each module. The only problem with this is
  //   it won't create queues for TopIO since that isn't a real module
  private def generateSimQueues(wrappers: Seq[Module]): SimQMap = {
    def rec(wrappers: Seq[Module], map: SimQMap): SimQMap = {
      if (wrappers.isEmpty) map
      else {
        val w = wrappers.head
        val name = unwrapName(w.name)
        val newMap = (w.ports filter(isSimPort) map { port =>
          (splitSimPort(port) map { field =>
            val (src, dst) = if (field.dir == Default) (name, port.name) else (port.name, name)
            if (map.contains((src, dst))) SimQMap
            else Map((src, dst) -> buildSimQueue(queueName(src, dst), getHostBits(field).tpe))
          }).flatten.toMap
        }).flatten.toMap
        rec(wrappers.tail, map ++ newMap)
      }
    }
    rec(wrappers, SimQMap)
  }

  // ********** generateSimTop **********
  // Creates the Simulation Top module where all sim modules and sim queues are instantiated and connected
  private def transformTopIO(ports: Seq[Port]): Seq[Port] = {
    val noClock = ports filter (_.tpe != ClockType)
    val inputs  = noClock filter (_.dir == Input) map (_.toField.flip) // Flip because wrapping port is input
    val outputs = noClock filter (_.dir == Output) map (_.toField)

    Seq(Port(NoInfo, "io", Output, BundleType(Seq(Field("hostIn", Reverse, genHostDecoupled(inputs)),
                                                  Field("hostOut", Default, genHostDecoupled(outputs))))))
  }
  private def generateSimTop(wrappers: Seq[Module], simQueues: SimQMap, portMap: PortMap, rtlTop: Module): Module = {
    val insts = (wrappers map { m => DefInst(NoInfo, instName(m.name), buildExp(m.name)) }) ++
                (simQueues.values map { m => DefInst(NoInfo, instName(m.name), buildExp(m.name)) })
    val connectClocks = (wrappers ++ simQueues.values) map { m => 
      Connect(NoInfo, buildExp(Seq(instName(m.name), hostClock.name)), buildExp(hostClock.name)) 
    }
    val connectResets = (wrappers ++ simQueues.values) map { m =>
      Connect(NoInfo, buildExp(Seq(instName(m.name), hostReset.name)), buildExp(hostReset.name))
    }
    // Connect queues to simulation modules (excludes IO)
    val connectQueues = (simQueues map { case ((src, dst), queue) =>
      (if (src == "topIO") Seq()
       else Seq(BulkConnect(NoInfo, buildExp(Seq(instName(queue.name), "io", "enq")), 
                                    buildExp(Seq(instName(wrapName(src)), dst, "hostOut"))))
      ) ++
      (if (dst == "topIO") Seq()
       else Seq(BulkConnect(NoInfo, buildExp(Seq(instName(wrapName(dst)), src, "hostIn")),
                                    buildExp(Seq(instName(queue.name), "io", "deq"))))
      )
    }).flatten
    // Connect IO queues, Src means input, Dst means output (ie. the outside word is the Src or Dst)
    val ioSrcQueues = (simQueues filter {case ((src, dst), _) => src == "topIO"} map {case (_, queue) => queue}).toSeq
    val ioDstQueues = (simQueues filter {case ((src, dst), _) => dst == "topIO"} map {case (_, queue) => queue}).toSeq
    val ioSrcSignals = rtlTop.ports filter (sig => sig.tpe != ClockType && sig.dir == Input) map (_.name)
    val ioDstSignals = rtlTop.ports filter (sig => sig.tpe != ClockType && sig.dir == Output) map (_.name)

    val ioSrcQueueConnect = if (ioSrcQueues.length > 0) {
      val readySignals = ioSrcQueues map (queue => buildExp(Seq(instName(queue.name), "io", "enq", hostReady.name)))
      val validSignals = ioSrcQueues map (queue => buildExp(Seq(instName(queue.name), "io", "enq", hostValid.name)))

      (ioSrcSignals map { sig => 
        (portMap(sig) map { dst => 
          Connect(NoInfo, buildExp(Seq(instName(queueName("topIO", dst)), "io", "enq", "hostBits", sig)),
                          buildExp(Seq("io", "hostIn", "hostBits", sig)))
        })
      }).flatten ++
      (validSignals map (sig => Connect(NoInfo, buildExp(sig), buildExp(Seq("io", "hostIn", hostValid.name))))) :+
      Connect(NoInfo, buildExp(Seq("io", "hostIn", hostReady.name)), genPrimopReduce(And, readySignals))
    } else Seq(EmptyStmt)

    val ioDstQueueConnect = if (ioDstQueues.length > 0) {
      val readySignals = ioDstQueues map (queue => buildExp(Seq(instName(queue.name), "io", "deq", hostReady.name)))
      val validSignals = ioDstQueues map (queue => buildExp(Seq(instName(queue.name), "io", "deq", hostValid.name)))

      (ioDstSignals map { sig => 
        (portMap(sig) map { src => 
          Connect(NoInfo, buildExp(Seq("io", "hostOut", "hostBits", sig)),
                          buildExp(Seq(instName(queueName(src, "topIO")), "io", "deq", "hostBits", sig)))
        })
      }).flatten ++
      (readySignals map (sig => Connect(NoInfo, buildExp(sig), buildExp(Seq("io", "hostOut", hostReady.name))))) :+
      Connect(NoInfo, buildExp(Seq("io", "hostOut", hostValid.name)), genPrimopReduce(And, validSignals))
    } else Seq(EmptyStmt)

    val stmts = Block(insts ++ connectClocks ++ connectResets ++ connectQueues ++ ioSrcQueueConnect ++ ioDstQueueConnect)
    val ports = Seq(hostClock, hostReset) ++ transformTopIO(rtlTop.ports)
    Module(NoInfo, "SimTop", ports, stmts)
  }

  // ********** transform **********
  // Perform FAME-1 Transformation for MIDAS
  def transform(c: Circuit): Circuit = {
    // We should check that the invariants mentioned above are true
    val nameToModule = (c.modules map (m => m.name -> m))(collection.breakOut): Map[String, Module] 
    val top = nameToModule(c.name)

    val rtlModules = c.modules filter (_.name != top.name) map (transformRTL)

    val insts = getDefInsts(top)

    val portConn = findPortConn(top, insts)

    // Check that port Connections include all ports for each instance?

    val simWrappers = insts map (inst => genWrapperModule(inst, portConn(inst.name)))

    val simQueues = generateSimQueues(simWrappers)

    // Remove duplicate simWrapper and simQueue modules?

    val simTop = generateSimTop(simWrappers, simQueues, portConn("topIO"), top)

    val modules = rtlModules ++ simWrappers ++ simQueues.values.toSeq ++ Seq(simTop)

    Circuit(c.info, simTop.name, modules)
  }

}
