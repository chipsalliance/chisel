package chisel3.util.experimental

import chisel3.InstanceId
import chisel3.experimental.{ChiselAnnotation, RunFirrtlTransform, annotate}
import firrtl.{CircuitState, LowForm, RenameMap, Transform}
import firrtl.annotations.{Annotation, CompleteTarget}
import java.io.FileWriter

import firrtl.options.TargetDirAnnotation

object Tracker {

  /** Tracks how the given signal/module's name changes throughout compilation
    * Changes are written to the specified file, placed in the target output directory
    * If no file is specified, changes are written to "Trackers.txt"
    *
    * @param id signal/module whose name to track
    * @param file optional filename specifier
    */
  def track(id: InstanceId,
            file: Option[String] = None): Unit = {

    val annotations =
      Seq(new ChiselAnnotation with RunFirrtlTransform {
        def toFirrtl = TrackerAnnotation(Some(id.toTarget), None, None, file)
        def transformClass: Class[TrackerWriter] = classOf[TrackerWriter] })

    annotations.map(annotate(_))
  }
}

/** Tracks the target throughout FIRRTL compilation
  *
  * @param target Current name of the target
  * @param from Previous name of the target, if exists (e.g. not the original name)
  * @param cause Cause for the change. For now it is always None, but in the future may contain transform name
  * @param file Optional name of the file to write changes to
  */
case class TrackerAnnotation(target: Option[CompleteTarget],
                             from: Option[TrackerAnnotation] = None,
                             cause: Option[String] = None,
                             file: Option[String] = None) extends Annotation {

  override def update(renames: RenameMap): Seq[TrackerAnnotation] = {
    if(target.isDefined) {
      renames.get(target.get) match {
        case None => Seq(this)
        case Some(Seq()) => Seq(TrackerAnnotation(None, Some(this), None, file))
        case Some(targets) =>
          //TODO: Add cause of renaming, requires FIRRTL change to RenameMap
          targets.map { t => TrackerAnnotation(Some(t), Some(this), None, file)}
      }
    } else Seq(this)
  }

  private def expand(stringBuilder: StringBuilder): StringBuilder = {
    if(target.isDefined) {
      stringBuilder.append(s"${target.get.serialize}")
    } else {
      stringBuilder.append(s"<DELETED>")
    }
    if(from.isDefined) {
      val arrow = cause.map("(" + _ + ")").getOrElse("")
      stringBuilder.append(s" <-$arrow- ")
      from.get.expand(stringBuilder)
    }
    stringBuilder
  }

  override def serialize: String = expand(new StringBuilder()).toString
}

/** Writes [[TrackerAnnotation]] history to a file
  *
  * Does not modify circuit
  * Is currently scheduled via its LowForm
  *
  * TODO: Schedule using Dependency API to ensure it is the final transform
  */
class TrackerWriter extends Transform {
  override def inputForm = LowForm

  override def outputForm = LowForm

  override def execute(state: CircuitState): CircuitState = {
    val tracks = state.annotations.collect { case t: TrackerAnnotation => t}
    val targetDir = state.annotations.collectFirst {
      case d: TargetDirAnnotation => d.directory
    }.getOrElse(".")

    logger.info(s"Writing trackers to directory $targetDir.")

    val trackMap = tracks.groupBy { t => t.file }
    trackMap.foreach {
      case (file, trackers) =>
        val writer = new FileWriter(targetDir + "/" + file.getOrElse("Trackers.txt"))
        trackers.foreach { t => writer.write(t.serialize + "\n") }
        writer.write("\n")
        writer.close()
    }

    state
  }
}
