// See LICENSE for license details.
package firrtl.transforms
package TopWiring

import firrtl._
import firrtl.ir._
import firrtl.passes.{Pass,
      InferTypes,
      ResolveKinds,
      ResolveGenders
      }
import firrtl.annotations._
import firrtl.Mappers._
import firrtl.graph._

import java.io._
import scala.io.Source
import collection.mutable

/** Annotation for optional output files, and what directory to put those files in (absolute path) **/
case class TopWiringOutputFilesAnnotation(dirName: String,
                                          outputFunction: (String,Seq[((ComponentName, Type, Boolean,
                                                                         Seq[String],String), Int)],
                                                           CircuitState) => CircuitState) extends NoTargetAnnotation

/** Annotation for indicating component to be wired, and what prefix to add to the ports that are generated */
case class TopWiringAnnotation(target: ComponentName, prefix: String) extends
    SingleTargetAnnotation[ComponentName] {
  def duplicate(n: ComponentName) = this.copy(target = n)
}


/** Punch out annotated ports out to the toplevel of the circuit.
    This also has an option to pass a function as a parmeter to generate
    custom output files as a result of the additional ports
  * @note This *does* work for deduped modules
  */
class TopWiringTransform extends Transform {
  def inputForm: CircuitForm = LowForm
  def outputForm: CircuitForm = LowForm

  type InstPath = Seq[String]

  /** Get the names of the targets that need to be wired */
  private def getSourceNames(state: CircuitState): Map[ComponentName, String] = {
    state.annotations.collect { case TopWiringAnnotation(srcname,prefix) =>
                                  (srcname -> prefix) }.toMap.withDefaultValue("")
  }


  /** Get the names of the modules which include the  targets that need to be wired */
  private def getSourceModNames(state: CircuitState): Seq[String] = {
    state.annotations.collect { case TopWiringAnnotation(ComponentName(_,ModuleName(srcmodname, _)),_) => srcmodname }
  }



  /** Get the Type of each wire to be connected
    *
    * Find the definition of each wire in sourceList, and get the type and whether or not it's a port
    * Update the results in sourceMap
    */
  private def getSourceTypes(sourceList: Map[ComponentName, String],
                     sourceMap: mutable.Map[String, Seq[(ComponentName, Type, Boolean, InstPath, String)]],
                     currentmodule: ModuleName, state: CircuitState)(s: Statement): Statement = s match {
    // If target wire, add name and size to to sourceMap
    case w: IsDeclaration =>
      if (sourceList.keys.toSeq.contains(ComponentName(w.name, currentmodule))) {
          val (isport, tpe, prefix) = w match {
            case d: DefWire => (false, d.tpe, sourceList(ComponentName(w.name,currentmodule)))
            case d: DefNode => (false, d.value.tpe, sourceList(ComponentName(w.name,currentmodule)))
            case d: DefRegister => (false, d.tpe, sourceList(ComponentName(w.name,currentmodule)))
            case d: Port => (true, d.tpe, sourceList(ComponentName(w.name,currentmodule)))
            case _ => throw new Exception(s"Cannot wire this type of declaration! ${w.serialize}")
          }
          val name = w.name
          sourceMap.get(currentmodule.name) match {
            case Some(xs:Seq[(ComponentName, Type, Boolean, InstPath, String)]) =>
              sourceMap.update(currentmodule.name, xs :+
                 (ComponentName(w.name,currentmodule), tpe, isport ,Seq[String](w.name), prefix))
            case None =>
              sourceMap(currentmodule.name) = Seq((ComponentName(w.name,currentmodule),
                                                   tpe, isport ,Seq[String](w.name), prefix))
          }
      }
      w // Return argument unchanged (ok because DefWire has no Statement children)
    // If not, apply to all children Statement
    case _ => s map getSourceTypes(sourceList, sourceMap, currentmodule, state)
  }



  /** Get the Type of each port to be connected
    *
    * Similar to getSourceTypes, but specifically for ports since they are not found in statements.
    * Find the definition of each port in sourceList, and get the type and whether or not it's a port
    * Update the results in sourceMap
    */
  private def getSourceTypesPorts(sourceList: Map[ComponentName, String], sourceMap: mutable.Map[String,
                          Seq[(ComponentName, Type, Boolean, InstPath, String)]],
                          currentmodule: ModuleName, state: CircuitState)(s: Port): CircuitState = s match {
    // If target port, add name and size to to sourceMap
    case w: IsDeclaration =>
      if (sourceList.keys.toSeq.contains(ComponentName(w.name, currentmodule))) {
          val (isport, tpe, prefix) = w match {
            case d: Port => (true, d.tpe, sourceList(ComponentName(w.name,currentmodule)))
            case _ => throw new Exception(s"Cannot wire this type of declaration! ${w.serialize}")
          }
          val name = w.name
          sourceMap.get(currentmodule.name) match {
            case Some(xs:Seq[(ComponentName, Type, Boolean, InstPath, String)]) =>
                sourceMap.update(currentmodule.name, xs :+
                  (ComponentName(w.name,currentmodule), tpe, isport ,Seq[String](w.name), prefix))
            case None =>
                sourceMap(currentmodule.name) = Seq((ComponentName(w.name,currentmodule),
                                                     tpe, isport ,Seq[String](w.name), prefix))
          }
      }
      state // Return argument unchanged (ok because DefWire has no Statement children)
    // If not, apply to all children Statement
    case _ => state
  }


  /** Create a map of Module name to target wires under this module
    *
    * These paths are relative but cross module (they refer down through instance hierarchy)
    */
  private def getSourcesMap(state: CircuitState): Map[String,Seq[(ComponentName, Type, Boolean, InstPath, String)]] = {
    val sSourcesModNames = getSourceModNames(state)
    val sSourcesNames = getSourceNames(state)
    val instGraph = new firrtl.analyses.InstanceGraph(state.circuit)
    val cMap = instGraph.getChildrenInstances.map{ case (m, wdis) =>
        (m -> wdis.map{ case wdi => (wdi.name, wdi.module) }.toSeq) }.toMap
    val topSort = instGraph.moduleOrder.reverse

    // Map of component name to relative instance paths that result in a debug wire
    val sourcemods: mutable.Map[String, Seq[(ComponentName, Type, Boolean, InstPath, String)]] =
      mutable.Map(sSourcesModNames.map(_ -> Seq()): _*)

    state.circuit.modules.foreach { m => m map
      getSourceTypes(sSourcesNames, sourcemods, ModuleName(m.name, CircuitName(state.circuit.main)) , state) }
    state.circuit.modules.foreach { m => m.ports.foreach {
       p => Seq(p) map
          getSourceTypesPorts(sSourcesNames, sourcemods, ModuleName(m.name, CircuitName(state.circuit.main)) , state) }}

    for (mod <- topSort) {
      val seqChildren: Seq[(ComponentName,Type,Boolean,InstPath,String)] = cMap(mod.name).flatMap {
        case (inst, module) =>
          sourcemods.get(module).map( _.map { case (a,b,c,path,p) => (a,b,c, inst +: path, p)})
      }.flatten
      if (seqChildren.nonEmpty) {
        sourcemods(mod.name) = sourcemods.getOrElse(mod.name, Seq()) ++ seqChildren
      }
    }

    sourcemods.toMap
  }



  /** Process a given DefModule
    *
    * For Modules that contain or are in the parent hierarchy to modules containing target wires
    * 1. Add ports for each target wire this module is parent to
    * 2. Connect these ports to ports of instances that are parents to some number of target wires
    */
  private def onModule(sources: Map[String, Seq[(ComponentName, Type, Boolean, InstPath, String)]],
                       portnamesmap : mutable.Map[String,String],
                       instgraph : firrtl.analyses.InstanceGraph,
                       namespacemap : Map[String, Namespace])
                      (module: DefModule): DefModule = {
    val namespace = namespacemap(module.name)
    sources.get(module.name) match {
      case Some(p) =>
        val newPorts = p.map{ case (ComponentName(cname,_), tpe, _ , path, prefix) => {
              val newportname = portnamesmap.get(prefix + path.mkString("_")) match {
                case Some(pn) => pn
                 case None => {
                    val npn = namespace.newName(prefix + path.mkString("_"))
                    portnamesmap(prefix + path.mkString("_")) = npn
                    npn
                 }
              }
              Port(NoInfo, newportname, Output, tpe)
        } }

        // Add connections to Module
        module match {
          case m: Module =>
            val connections: Seq[Connect] = p.map { case (ComponentName(cname,_), _, _ , path, prefix) =>
                val modRef = portnamesmap.get(prefix + path.mkString("_")) match {
                       case Some(pn) => WRef(pn)
                       case None => {
                          portnamesmap(prefix + path.mkString("_")) = namespace.newName(prefix + path.mkString("_"))
                          WRef(portnamesmap(prefix + path.mkString("_")))
                       }
                }
                path.size match {
                   case 1 => {
                       val leafRef = WRef(path.head.mkString(""))
                       Connect(NoInfo, modRef, leafRef)
                   }
                   case _ =>  {
                       val instportname =  portnamesmap.get(prefix + path.tail.mkString("_")) match {
                           case Some(ipn) => ipn
                           case None => {
                             val instmod = instgraph.getChildrenInstances(module.name).collectFirst {
                                 case wdi if wdi.name == path.head => wdi.module}.get
                             val instnamespace = namespacemap(instmod)
                             portnamesmap(prefix + path.tail.mkString("_")) =
                               instnamespace.newName(prefix + path.tail.mkString("_"))
                             portnamesmap(prefix + path.tail.mkString("_"))
                           }
                       }
                       val instRef = WSubField(WRef(path.head), instportname)
                       Connect(NoInfo, modRef, instRef)
                  }
                }
            }
            m.copy(ports = m.ports ++ newPorts, body = Block(Seq(m.body) ++ connections ))
          case e: ExtModule =>
            e.copy(ports = e.ports ++ newPorts)
      }
      case None => module // unchanged if no paths
    }
  }

  /** Run passes to fix up the circuit of making the new connections  */
  private def fixupCircuit(circuit: Circuit): Circuit = {
    val passes = Seq(
      InferTypes,
      ResolveKinds,
      ResolveGenders
    )
    passes.foldLeft(circuit) { case (c: Circuit, p: Pass) => p.run(c) }
  }


  /** Dummy function that is currently unused. Can be used to fill an outputFunction requirment in the future  */
  def topWiringDummyOutputFilesFunction(dir: String,
                                        mapping: Seq[((ComponentName, Type, Boolean, InstPath, String), Int)],
                                        state: CircuitState): CircuitState = {
     state
  }


  def execute(state: CircuitState): CircuitState = {

    val outputTuples: Seq[(String,
                          (String,Seq[((ComponentName, Type, Boolean, InstPath, String), Int)],
                                        CircuitState) => CircuitState)] = state.annotations.collect {
         case TopWiringOutputFilesAnnotation(td,of) => (td, of) }
    // Do actual work of this transform
    val sources = getSourcesMap(state)
    val (nstate, nmappings) = if (sources.nonEmpty) {
      val portnamesmap: mutable.Map[String,String] = mutable.Map()
      val instgraph = new firrtl.analyses.InstanceGraph(state.circuit)
      val namespacemap = state.circuit.modules.map{ case m => (m.name -> Namespace(m)) }.toMap
      val modulesx = state.circuit.modules map onModule(sources, portnamesmap, instgraph, namespacemap)
      val newCircuit = state.circuit.copy(modules = modulesx)
      val fixedCircuit = fixupCircuit(newCircuit)
      val mappings = sources(state.circuit.main).zipWithIndex
      (state.copy(circuit = fixedCircuit), mappings)
    }
    else { (state, List.empty) }
    //Generate output files based on the mapping.
    outputTuples.map { case (dir, outputfunction) => outputfunction(dir, nmappings, nstate) }
    nstate
  }
}
