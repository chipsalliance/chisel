// See LICENSE for license details.

package firrtl.transforms

import firrtl._
import firrtl.annotations._
import firrtl.annotations.TargetToken.OfModule
import firrtl.analyses.InstanceKeyGraph
import firrtl.analyses.InstanceKeyGraph.InstanceKey
import firrtl.options.Dependency
import firrtl.stage.Forms
import firrtl.graph.DiGraph

import java.io.{File, FileWriter}

/** Marks modules as "must deduplicate" */
case class MustDeduplicateAnnotation(modules: Seq[IsModule]) extends Annotation {

  def update(renames: RenameMap): Seq[MustDeduplicateAnnotation] = {
    val newModules: Seq[IsModule] = modules.flatMap { m =>
      renames.get(m) match {
        case None        => Seq(m)
        case Some(Seq()) => Seq()
        case Some(Seq(one: IsModule)) => Seq(one)
        case Some(many) =>
          val msg = "Something went wrong! This anno's targets should only rename to IsModules! " +
            s"Got: ${m.serialize} -> ${many.map(_.serialize).mkString(", ")}"
          throw new Exception(msg)
      }
    }
    if (newModules.isEmpty) Seq() else Seq(this.copy(newModules))
  }
}

/** Specifies the directory where errors for modules that "must deduplicate" will be reported */
case class MustDeduplicateReportDirectory(directory: String) extends NoTargetAnnotation

object MustDeduplicateTransform {
  sealed trait DedupFailureCandidate {
    def message: String
    def modules: Seq[OfModule]
  }
  case class LikelyShouldMatch(a: OfModule, b: OfModule) extends DedupFailureCandidate {
    def message: String = s"Modules '${a.value}' and '${b.value}' likely should dedup but do not."
    def modules = Seq(a, b)
  }
  object DisjointChildren {
    sealed trait Reason
    case object Left extends Reason
    case object Right extends Reason
    case object Both extends Reason
  }
  import DisjointChildren._
  case class DisjointChildren(a: OfModule, b: OfModule, reason: Reason) extends DedupFailureCandidate {
    def message: String = {
      def helper(x: OfModule, y: OfModule): String = s"'${x.value}' contains instances not found in '${y.value}'"
      val why = reason match {
        case Left  => helper(a, b)
        case Right => helper(b, a)
        case Both  => s"${helper(a, b)} and ${helper(b, a)}"
      }
      s"Modules '${a.value}' and '${b.value}' cannot be deduplicated because $why."
    }
    def modules = Seq(a, b)
  }

  final class DeduplicationFailureException(msg: String) extends FirrtlUserException(msg)

  case class DedupFailure(
    shouldDedup:  Seq[OfModule],
    relevantMods: Set[OfModule],
    candidates:   Seq[DedupFailureCandidate])

  /** Reports deduplication failures two Modules
    *
    * @return (Set of Modules that only appear in one hierarchy or the other, candidate pairs of Module names)
    */
  def findDedupFailures(shouldDedup: Seq[OfModule], graph: InstanceKeyGraph): DedupFailure = {
    val instLookup = graph.getChildInstances.toMap
    def recurse(a: OfModule, b: OfModule): Seq[DedupFailureCandidate] = {
      val as = instLookup(a.value)
      val bs = instLookup(b.value)
      if (as.length != bs.length) {
        val aa = as.toSet
        val bb = bs.toSet
        val reason = (aa.diff(bb).nonEmpty, bb.diff(aa).nonEmpty) match {
          case (true, true)  => Both
          case (true, false) => Left
          case (false, true) => Right
          case _             => Utils.error("Impossible!")
        }
        Seq(DisjointChildren(a, b, reason))
      } else {
        val fromChildren = as.zip(bs).flatMap {
          case (ax, bx) => recurse(ax.OfModule, bx.OfModule)
        }
        if (fromChildren.nonEmpty) fromChildren
        else if (a != b) Seq(LikelyShouldMatch(a, b))
        else Nil
      }
    }

    val allMismatches = {
      // Recalculating this every time is a little wasteful, but we're on a failure path anyway
      val digraph = graph.graph.transformNodes(_.OfModule)
      val froms = shouldDedup.map(x => digraph.reachableFrom(x) + x)
      val union = froms.reduce(_ union _)
      val intersection = froms.reduce(_ intersect _)
      union.diff(intersection)
    }.toSet
    val pairs = shouldDedup.tail.map(n => (shouldDedup.head, n))
    val candidates = pairs.flatMap { case (a, b) => recurse(a, b) }
    DedupFailure(shouldDedup, allMismatches, candidates)
  }

  // Find the minimal number of vertices in the graph to show paths from "mustDedup" to failure
  // candidates and their context (eg. children for DisjoinChildren)
  private def findNodesToKeep(failure: DedupFailure, graph: DiGraph[String]): collection.Set[String] = {
    val shouldDedup = failure.shouldDedup.map(_.value).toSet
    val nodeOfInterest: Set[String] =
      shouldDedup ++ failure.candidates.flatMap {
        case LikelyShouldMatch(OfModule(a), OfModule(b)) => Seq(a, b)
        case DisjointChildren(OfModule(a), OfModule(b), _) =>
          Seq(a, b) ++ graph.getEdges(a) ++ graph.getEdges(b)
      }
    // Depth-first search looking for relevant nodes
    def dfs(node: String): collection.Set[String] = {
      val deeper = graph.getEdges(node).flatMap(dfs)
      if (deeper.nonEmpty || nodeOfInterest(node)) deeper + node else deeper
    }
    shouldDedup.flatMap(dfs)
  }

  /** Turn a [[DedupFailure]] into a pretty graph for visualization
    *
    * @param failure Failure to visualize
    * @param graph DiGraph of module names (no instance information)
    */
  def makeDedupFailureDiGraph(failure: DedupFailure, graph: DiGraph[String]): DiGraph[String] = {
    // Recalculating this every time is a little wasteful, but we're on a failure path anyway
    // Lookup the parent Module name of any Module
    val getParents: String => Seq[String] =
      graph.reverse.getEdgeMap
        .mapValues(_.toSeq)

    val candidates = failure.candidates
    val shouldDedup = failure.shouldDedup.map(_.value)
    val shouldDedupSet = shouldDedup.toSet
    val mygraph = {
      // Create a graph of paths from "shouldDedup" nodes to the candidates
      // rooted at the "shouldDedup" nodes
      val nodesToKeep = findNodesToKeep(failure, graph)
      graph.subgraph(nodesToKeep) +
        // Add fake nodes to represent parents of the "shouldDedup" nodes
        DiGraph(shouldDedup.map(n => getParents(n).mkString(", ") -> n): _*)
    }
    // Gather candidate modules and assign indices for reference
    val candidateIdx: Map[String, Int] =
      candidates.zipWithIndex.flatMap { case (c, idx) => c.modules.map(_.value -> idx) }.toMap
    // Now mark the graph for modules of interest
    val markedGraph = mygraph.transformNodes { n =>
      val next = if (shouldDedupSet(n)) s"($n)" else n
      candidateIdx
        .get(n)
        .map(i => s"$next [$i]")
        .getOrElse(next)
    }
    markedGraph
  }
}

/** Checks for modules that have been marked as "must deduplicate"
  *
  * In cases where marked modules did not deduplicate, this transform attempts to provide context on
  * what went wrong for debugging.
  */
class MustDeduplicateTransform extends Transform with DependencyAPIMigration {
  import MustDeduplicateTransform._

  override def prerequisites = Seq(Dependency[DedupModules])

  // Make this run as soon after Dedup as possible
  override def optionalPrerequisiteOf = (Forms.MidForm.toSet -- Forms.HighForm).toSeq

  override def invalidates(a: Transform) = false

  def execute(state: CircuitState): CircuitState = {

    lazy val igraph = InstanceKeyGraph(state.circuit)

    val dedupFailures: Seq[DedupFailure] =
      state.annotations.flatMap {
        case MustDeduplicateAnnotation(mods) =>
          val moduleNames = mods.map(_.leafModule).distinct
          if (moduleNames.size <= 1) None
          else {
            val modNames = moduleNames.map(OfModule)
            Some(findDedupFailures(modNames, igraph))
          }
        case _ => None
      }
    if (dedupFailures.nonEmpty) {
      val modgraph = igraph.graph.transformNodes(_.module)
      // Create and log reports
      val reports = dedupFailures.map {
        case fail @ DedupFailure(shouldDedup, _, candidates) =>
          val graph = makeDedupFailureDiGraph(fail, modgraph).prettyTree()
          val mods = shouldDedup.map("'" + _.value + "'").mkString(", ")
          val msg =
            s"""===== $mods are marked as "must deduplicate", but did not deduplicate. =====
               |$graph
               |Failure candidates:
               |${candidates.zipWithIndex.map { case (c, i) => s" - [$i] " + c.message }.mkString("\n")}
               |""".stripMargin
          logger.error(msg)
          msg
      }

      // Write reports and modules to disk
      val dirName = state.annotations.collectFirst { case MustDeduplicateReportDirectory(dir) => dir }
        .getOrElse("dedup_failures")
      val dir = new File(dirName)
      logger.error(s"Writing error report(s) to ${dir}...")
      FileUtils.makeDirectory(dir.toString)
      for ((report, idx) <- reports.zipWithIndex) {
        val f = new File(dir, s"report_$idx.rpt")
        logger.error(s"Writing $f...")
        val fw = new FileWriter(f)
        fw.write(report)
        fw.close()
      }

      val modsDir = new File(dir, "modules")
      FileUtils.makeDirectory(modsDir.toString)
      logger.error(s"Writing relevant modules to $modsDir...")
      val relevantModule = dedupFailures.flatMap(_.relevantMods.map(_.value)).toSet
      for (mod <- state.circuit.modules if relevantModule(mod.name)) {
        val fw = new FileWriter(new File(modsDir, s"${mod.name}.fir"))
        fw.write(mod.serialize)
        fw.close()
      }

      val msg = s"Modules marked 'must deduplicate' failed to deduplicate! See error reports in $dirName"
      throw new DeduplicationFailureException(msg)
    }
    state
  }
}
