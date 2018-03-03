// See LICENSE for license details.

package firrtl.passes
package wiring

import firrtl._
import firrtl.ir._
import firrtl.Utils._
import firrtl.Mappers._
import scala.collection.mutable
import firrtl.annotations._
import firrtl.annotations.AnnotationUtils._
import firrtl.analyses.InstanceGraph
import WiringUtils._

/** A data store of one sink--source wiring relationship */
case class WiringInfo(source: ComponentName, sinks: Seq[Named], pin: String)

/** A data store of wiring names */
case class WiringNames(compName: String, source: String, sinks: Seq[Named],
                       pin: String)

/** Pass that computes and applies a sequence of wiring modifications
  *
  * @constructor construct a new Wiring pass
  * @param wiSeq the [[WiringInfo]] to apply
  */
class Wiring(wiSeq: Seq[WiringInfo]) extends Pass {
  def run(c: Circuit): Circuit = analyze(c)
    .foldLeft(c){
      case (cx, (tpe, modsMap)) => cx.copy(
        modules = cx.modules map onModule(tpe, modsMap)) }

  /** Converts multiple units of wiring information to module modifications */
  private def analyze(c: Circuit): Seq[(Type, Map[String, Modifications])] = {

    val names = wiSeq
      .map ( wi => (wi.source, wi.sinks, wi.pin) match {
              case (ComponentName(comp, ModuleName(source,_)), sinks, pin) =>
                WiringNames(comp, source, sinks, pin) })

    val portNames = mutable.Seq.fill(names.size)(Map[String, String]())
    c.modules.foreach{ m =>
      val ns = Namespace(m)
      names.zipWithIndex.foreach{ case (WiringNames(c, so, si, p), i) =>
        portNames(i) = portNames(i) +
        ( m.name -> {
           if (si.exists(getModuleName(_) == m.name)) ns.newName(p)
           else ns.newName(tokenize(c) filterNot ("[]." contains _) mkString "_")
         })}}

    val iGraph = new InstanceGraph(c)
    names.zip(portNames).map{ case(WiringNames(comp, so, si, _), pn) =>
      computeModifications(c, iGraph, comp, so, si, pn) }
  }

  /** Converts a single unit of wiring information to module modifications
    *
    * @param c the circuit that will be modified
    * @param iGraph an InstanceGraph representation of the circuit
    * @param compName the name of a component
    * @param source the name of the source component
    * @param sinks a list of sink components/modules that the source
    * should be connected to
    * @param portNames a mapping of module names to new ports/wires
    * that should be generated if needed
    *
    * @return a tuple of the component type and a map of module names
    * to pending modifications
    */
  private def computeModifications(c: Circuit,
                                   iGraph: InstanceGraph,
                                   compName: String,
                                   source: String,
                                   sinks: Seq[Named],
                                   portNames: Map[String, String]):
      (Type, Map[String, Modifications]) = {

    val sourceComponentType = getType(c, source, compName)
    val sinkComponents: Map[String, Seq[String]] = sinks
      .collect{ case ComponentName(c, ModuleName(m, _)) => (c, m) }
      .foldLeft(new scala.collection.immutable.HashMap[String, Seq[String]]){
        case (a, (c, m)) => a ++ Map(m -> (Seq(c) ++ a.getOrElse(m, Nil)) ) }

    // Determine "ownership" of sources to sinks via minimum distance
    val owners = sinksToSources(sinks, source, iGraph)

    // Determine port and pending modifications for all sink--source
    // ownership pairs
    val meta = new mutable.HashMap[String, Modifications]
      .withDefaultValue(Modifications())
    owners.foreach { case (sink, source) =>
      val lca = iGraph.lowestCommonAncestor(sink, source)

      // Compute metadata along Sink to LCA paths.
      sink.drop(lca.size - 1).sliding(2).toList.reverse.map {
        case Seq(WDefInstance(_,_,pm,_), WDefInstance(_,ci,cm,_)) =>
          val to = s"$ci.${portNames(cm)}"
          val from = s"${portNames(pm)}"
          meta(pm) = meta(pm).copy(
            addPortOrWire = Some((portNames(pm), DecWire)),
            cons = (meta(pm).cons :+ (to, from)).distinct
          )
          meta(cm) = meta(cm).copy(
            addPortOrWire = Some((portNames(cm), DecInput))
          )
        // Case where the sink is the LCA
        case Seq(WDefInstance(_,_,pm,_)) =>
          // Case where the source is also the LCA
          if (source.drop(lca.size).isEmpty) {
            meta(pm) = meta(pm).copy (
              addPortOrWire = Some((portNames(pm), DecWire))
            )
          } else {
            val WDefInstance(_,ci,cm,_) = source.drop(lca.size).head
            val to = s"${portNames(pm)}"
            val from = s"$ci.${portNames(cm)}"
            meta(pm) = meta(pm).copy(
              addPortOrWire = Some((portNames(pm), DecWire)),
              cons = (meta(pm).cons :+ (to, from)).distinct
            )
          }
      }

      // Compute metadata for the Sink
      sink.last match { case WDefInstance(_, _, m, _) =>
        if (sinkComponents.contains(m)) {
          val from = s"${portNames(m)}"
          sinkComponents(m).foreach( to =>
            meta(m) = meta(m).copy(
              cons = (meta(m).cons :+ (to, from)).distinct
            )
          )
        }
      }

      // Compute metadata for the Source
      source.last match { case WDefInstance(_, _, m, _) =>
        val to = s"${portNames(m)}"
        val from = compName
        meta(m) = meta(m).copy(
          cons = (meta(m).cons :+ (to, from)).distinct
        )
      }

      // Compute metadata along Source to LCA path
      source.drop(lca.size - 1).sliding(2).toList.reverse.map {
        case Seq(WDefInstance(_,_,pm,_), WDefInstance(_,ci,cm,_)) => {
          val to = s"${portNames(pm)}"
          val from = s"$ci.${portNames(cm)}"
          meta(pm) = meta(pm).copy(
            cons = (meta(pm).cons :+ (to, from)).distinct
          )
          meta(cm) = meta(cm).copy(
            addPortOrWire = Some((portNames(cm), DecOutput))
          )
        }
        // Case where the source is the LCA
        case Seq(WDefInstance(_,_,pm,_)) => {
          // Case where the sink is also the LCA. We do nothing here,
          // as we've created the connecting wire above
          if (sink.drop(lca.size).isEmpty) {
          } else {
            val WDefInstance(_,ci,cm,_) = sink.drop(lca.size).head
            val to = s"$ci.${portNames(cm)}"
            val from = s"${portNames(pm)}"
            meta(pm) = meta(pm).copy(
              cons = (meta(pm).cons :+ (to, from)).distinct
            )
          }
        }
      }
    }
    (sourceComponentType, meta.toMap)
  }

  /** Apply modifications to a module */
  private def onModule(t: Type, map: Map[String, Modifications])(m: DefModule) = {
    map.get(m.name) match {
      case None => m
      case Some(l) =>
        val defines = mutable.ArrayBuffer[Statement]()
        val connects = mutable.ArrayBuffer[Statement]()
        val ports = mutable.ArrayBuffer[Port]()
        l.addPortOrWire match {
          case None =>
          case Some((s, dt)) => dt match {
            case DecInput  => ports += Port(NoInfo, s, Input, t)
            case DecOutput => ports += Port(NoInfo, s, Output, t)
            case DecWire   => defines += DefWire(NoInfo, s, t)
          }
        }
        connects ++= (l.cons map { case ((l, r)) =>
          Connect(NoInfo, toExp(l), toExp(r))
        })
        m match {
          case Module(i, n, ps, body) =>
            val stmts = body match {
              case Block(sx) => sx
              case s => Seq(s)
            }
            Module(i, n, ps ++ ports, Block(defines ++ stmts ++ connects))
          case ExtModule(i, n, ps, dn, p) => ExtModule(i, n, ps ++ ports, dn, p)
        }
    }
  }
}
