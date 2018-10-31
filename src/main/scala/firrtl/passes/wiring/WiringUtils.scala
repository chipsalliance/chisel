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
import firrtl.graph.DiGraph
import WiringUtils._

/** Declaration kind in lineage (e.g. input port, output port, wire)
  */
sealed trait DecKind
case object DecInput extends DecKind
case object DecOutput extends DecKind
case object DecWire extends DecKind

/** Store of pending wiring information for a Module */
case class Modifications(
  addPortOrWire: Option[(String, DecKind)] = None,
  cons: Seq[(String, String)] = Seq.empty) {

  override def toString: String = serialize("")

  def serialize(tab: String): String = s"""
   |$tab addPortOrWire: $addPortOrWire
   |$tab cons: $cons
   |""".stripMargin
}

/** A lineage tree representing the instance hierarchy in a design
  */
@deprecated("Use DiGraph/InstanceGraph", "1.1.1")
case class Lineage(
    name: String,
    children: Seq[(String, Lineage)] = Seq.empty,
    source: Boolean = false,
    sink: Boolean = false,
    sourceParent: Boolean = false,
    sinkParent: Boolean = false,
    sharedParent: Boolean = false,
    addPort: Option[(String, DecKind)] = None,
    cons: Seq[(String, String)] = Seq.empty) {

  def map(f: Lineage => Lineage): Lineage =
    this.copy(children = children.map{ case (i, m) => (i, f(m)) })

  override def toString: String = shortSerialize("")

  def shortSerialize(tab: String): String = s"""
    |$tab name: $name,
    |$tab children: ${children.map(c => tab + "   " + c._2.shortSerialize(tab + "    "))}
    |""".stripMargin

  def foldLeft[B](z: B)(op: (B, (String, Lineage)) => B): B =
    this.children.foldLeft(z)(op)

  def serialize(tab: String): String = s"""
    |$tab name: $name,
    |$tab source: $source,
    |$tab sink: $sink,
    |$tab sourceParent: $sourceParent,
    |$tab sinkParent: $sinkParent,
    |$tab sharedParent: $sharedParent,
    |$tab addPort: $addPort
    |$tab cons: $cons
    |$tab children: ${children.map(c => tab + "   " + c._2.serialize(tab + "    "))}
    |""".stripMargin
}

object WiringUtils {
  @deprecated("Use DiGraph/InstanceGraph", "1.1.1")
  type ChildrenMap = mutable.HashMap[String, Seq[(String, String)]]

  /** Given a circuit, returns a map from module name to children
    * instance/module names
    */
  @deprecated("Use DiGraph/InstanceGraph", "1.1.1")
  def getChildrenMap(c: Circuit): ChildrenMap = {
    val childrenMap = new ChildrenMap()
    def getChildren(mname: String)(s: Statement): Statement = s match {
      case s: WDefInstance =>
        childrenMap(mname) = childrenMap(mname) :+ (s.name, s.module)
        s
      case s: DefInstance =>
        childrenMap(mname) = childrenMap(mname) :+ (s.name, s.module)
        s
      case s => s map getChildren(mname)
    }
    c.modules.foreach{ m =>
      childrenMap(m.name) = Nil
      m map getChildren(m.name)
    }
    childrenMap
  }

  /** Returns a module's lineage, containing all children lineages as well
    */
  @deprecated("Use DiGraph/InstanceGraph", "1.1.1")
  def getLineage(childrenMap: ChildrenMap, module: String): Lineage =
    Lineage(module, childrenMap(module) map { case (i, m) => (i, getLineage(childrenMap, m)) } )

  /** Return a map of sink instances to source instances that minimizes
    * distance
    *
    * @param sinks a sequence of sink modules
    * @param source the source module
    * @param i a graph representing a circuit
    * @return a map of sink instance names to source instance names
    * @throws WiringException if a sink is equidistant to two sources
    */
  def sinksToSources(sinks: Seq[Named],
                     source: String,
                     i: InstanceGraph):
      Map[Seq[WDefInstance], Seq[WDefInstance]] = {
    val owners = new mutable.HashMap[Seq[WDefInstance], Vector[Seq[WDefInstance]]]
      .withDefaultValue(Vector())
    val queue = new mutable.Queue[Seq[WDefInstance]]
    val visited = new mutable.HashMap[Seq[WDefInstance], Boolean]
      .withDefaultValue(false)

    i.fullHierarchy.keys.filter { case WDefInstance(_,_,m,_) => m == source }
      .foreach( i.fullHierarchy(_)
                 .foreach { l =>
                   queue.enqueue(l)
                   owners(l) = Vector(l)
                 }
      )

    val sinkInsts = i.fullHierarchy.keys
      .filter { case WDefInstance(_, _, module, _) =>
        sinks.map(getModuleName(_)).contains(module) }
      .flatMap { k => i.fullHierarchy(k)          }
      .toSet

    /** If we're lucky and there is only one source, then that source owns
      * all sinks. If we're unlucky, we need to do a full (slow) BFS
      * to figure out who owns what. Currently, the BFS is not
      * performant.
      *
      * [todo] The performance of this will need to be improved.
      * Possible directions are that if we're purely source-under-sink
      * or sink-under-source, then ownership is trivially a mapping
      * down/up. Ownership seems to require a BFS if we have
      * sources/sinks not under sinks/sources.
      */
    if (queue.size == 1) {
      val u = queue.dequeue
      sinkInsts.foreach { v => owners(v) = Vector(u) }
    } else {
      while (queue.nonEmpty) {
        val u = queue.dequeue
        visited(u) = true

        val edges = (i.graph.getEdges(u.last).map(u :+ _).toVector :+ u.dropRight(1))

        // [todo] This is the critical section
        edges
          .filter( e => !visited(e) && e.nonEmpty )
          .foreach{ v =>
            owners(v) = owners(v) ++ owners(u)
            queue.enqueue(v)
          }
      }

      // Check that every sink has one unique owner. The only time that
      // this should fail is if a sink is equidistant to two sources.
      sinkInsts.foreach { s =>
        if (!owners.contains(s) || owners(s).size > 1) {
          throw new WiringException(
            s"Unable to determine source mapping for sink '${s.map(_.name)}'") }
      }
    }

    owners
      .collect { case (k, v) if sinkInsts.contains(k) => (k, v.flatten) }.toMap
  }

  /** Helper script to extract a module name from a named Module or Target */
  def getModuleName(n: Named): String = {
    n match {
      case ModuleName(m, _)                   => m
      case ComponentName(_, ModuleName(m, _)) => m
      case _ => throw new WiringException(
        "Only Components or Modules have an associated Module name")
    }
  }

  /** Determine the Type of a specific component
    *
    * @param c the circuit containing the target module
    * @param module the module containing the target component
    * @param comp the target component
    * @return the component's type
    * @throws WiringException if the module is not contained in the
    * circuit or if the component is not contained in the module
    */
  def getType(c: Circuit, module: String, comp: String): Type = {
    def getRoot(e: Expression): String = e match {
      case r: Reference => r.name
      case i: SubIndex => getRoot(i.expr)
      case a: SubAccess => getRoot(a.expr)
      case f: SubField => getRoot(f.expr)
    }
    val eComp = toExp(comp)
    val root = getRoot(eComp)
    var tpe: Option[Type] = None
    def getType(s: Statement): Statement = s match {
      case DefRegister(_, n, t, _, _, _) if n == root =>
        tpe = Some(t)
        s
      case DefWire(_, n, t) if n == root =>
        tpe = Some(t)
        s
      case WDefInstance(_, n, _, t) if n == root =>
        tpe = Some(t)
        s
      case DefNode(_, n, e) if n == root =>
        tpe = Some(e.tpe)
        s
      case sx: DefMemory if sx.name == root =>
        tpe = Some(MemPortUtils.memType(sx))
        sx
      case sx => sx map getType
    }
    val m = c.modules find (_.name == module) getOrElse {
      throw new WiringException(s"Must have a module named $module") }
    tpe = m.ports find (_.name == root) map (_.tpe)
    m match {
      case Module(i, n, ps, b) => getType(b)
      case e: ExtModule =>
    }
    tpe match {
      case None => throw new WiringException(s"Didn't find $comp in $module!")
      case Some(t) =>
        def setType(e: Expression): Expression = e map setType match {
          case ex: Reference => ex.copy(tpe = t)
          case ex: SubField => ex.copy(tpe = field_type(ex.expr.tpe, ex.name))
          case ex: SubIndex => ex.copy(tpe = sub_type(ex.expr.tpe))
          case ex: SubAccess => ex.copy(tpe = sub_type(ex.expr.tpe))
        }
        setType(eComp).tpe
    }
  }
}
