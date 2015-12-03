 
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

  // Find the top module of a firrtl.Circuit
  private def findTop(c: Circuit): Module = {
    val moduleMap = c.modules.map(m => m.name -> m)(collection.breakOut): Map[String, Module] 
    moduleMap(c.name)
  }

  private def genWrapperModuleName(m: Module): String = s"SimWrap_${m.name}"
  private def genWrapperModule(m: Module, connections: Seq[String]): Module = {
    Module(m.info, genWrapperModuleName(m), m.ports, m.stmt)
  }

  def transform(c: Circuit): Circuit = {
    //Circuit(c.info, c.name, c.modules.map(fame1Transform(_, c.name)))
    //println(s"In circuit ${c.name}, we have instances: ")
    //println(insts)
    val top = findTop(c)
    println("Top Module:")
    println(top.serialize)

    val insts = getDefInsts(top)
    println(s"In top module ${top.name}, we have instances: ")
    insts.foreach(i => println("  " + i.name))

    //val 

    //val simInsts = insts.map(convertInst(c, _))
    //println("Simulation instances: ")
    //simInsts.foreach{i =>
    //  println(s"${i.name} : ${i.module.name}")
    //  i.ports.foreach{p => 
    //    val endpoint = p.endpoint match {
    //      case UnknownSimInst => "?"
    //      case SimTopIO => "TopIO"
    //      case inst: SimInst => inst.name
    //    }
    //    println(s"  ${p.port.name} : ${p.port.dir.serialize} : ${endpoint}")
    //  }
    //}
    //println(simInsts)

    c
  }

}
