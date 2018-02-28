// See LICENSE for license details.

package firrtl.transforms

import java.io.{File, FileNotFoundException, FileOutputStream, PrintWriter}

import firrtl._
import firrtl.Utils.throwInternalError
import firrtl.annotations._

import scala.collection.mutable.ArrayBuffer

sealed trait BlackBoxHelperAnno extends Annotation

case class BlackBoxTargetDirAnno(targetDir: String) extends BlackBoxHelperAnno
    with NoTargetAnnotation {
  override def serialize: String = s"targetDir\n$targetDir"
}

case class BlackBoxResourceAnno(target: ModuleName, resourceId: String) extends BlackBoxHelperAnno
    with SingleTargetAnnotation[ModuleName] {
  def duplicate(n: ModuleName) = this.copy(target = n)
  override def serialize: String = s"resource\n$resourceId"
}

case class BlackBoxInlineAnno(target: ModuleName, name: String, text: String) extends BlackBoxHelperAnno
    with SingleTargetAnnotation[ModuleName] {
  def duplicate(n: ModuleName) = this.copy(target = n)
  override def serialize: String = s"inline\n$name\n$text"
}

/** Handle source for Verilog ExtModules (BlackBoxes)
  *
  * This transform handles the moving of Verilog source for black boxes into the
  * target directory so that it can be accessed by verilator or other backend compilers
  * While parsing it's annotations it looks for a BlackBoxTargetDir annotation that
  * will set the directory where the Verilog will be written.  This annotation is typically be
  * set by the execution harness, or directly in the tests
  */
class BlackBoxSourceHelper extends firrtl.Transform {
  private val DefaultTargetDir = new File(".")

  override def inputForm: CircuitForm = LowForm
  override def outputForm: CircuitForm = LowForm

  /** Collect BlackBoxHelperAnnos and and find the target dir if specified
    * @param annos a list of generic annotations for this transform
    * @return BlackBoxHelperAnnos and target directory
    */
  def collectAnnos(annos: Seq[Annotation]): (Set[BlackBoxHelperAnno], File) =
    annos.foldLeft((Set.empty[BlackBoxHelperAnno], DefaultTargetDir)) {
      case ((acc, tdir), anno) => anno match {
        case BlackBoxTargetDirAnno(dir) =>
          val targetDir = new File(dir)
          if (!targetDir.exists()) { FileUtils.makeDirectory(targetDir.getAbsolutePath) }
          (acc, targetDir)
        case a: BlackBoxHelperAnno => (acc + a, tdir)
        case _ => (acc, tdir)
      }
    }

  /**
    * write the verilog source for each annotation to the target directory
    * @note the state is not changed by this transform
    * @param state Input Firrtl AST
    * @return A transformed Firrtl AST
    */
  override def execute(state: CircuitState): CircuitState = {
    val (annos, targetDir) = collectAnnos(state.annotations)
    val fileList = annos.foldLeft(List.empty[String]) {
      case (fileList, anno) => anno match {
        case BlackBoxResourceAnno(_, resourceId) =>
          val name = resourceId.split("/").last
          val outFile = new File(targetDir, name)
          BlackBoxSourceHelper.copyResourceToFile(resourceId,outFile)
          outFile.getAbsolutePath +: fileList
        case BlackBoxInlineAnno(_, name, text) =>
          val outFile = new File(targetDir, name)
          val writer = new PrintWriter(outFile)
          writer.write(text)
          writer.close()
          outFile.getAbsolutePath +: fileList
        case _ => throwInternalError()
      }
    }
    // If we have BlackBoxes, generate the helper file.
    // If we don't, make sure it doesn't exist or we'll confuse downstream processing
    //  that triggers behavior on the existence of the file
    val helperFile = new File(targetDir, BlackBoxSourceHelper.FileListName)
    if (fileList.nonEmpty) {
      val writer = new PrintWriter(helperFile)
      writer.write(fileList.map { fileName => s"-v $fileName" }.mkString("\n"))
      writer.close()
    } else {
      helperFile.delete()
    }

    state
  }
}

object BlackBoxSourceHelper {
  val FileListName = "black_box_verilog_files.f"
  /**
    * finds the named resource and writes into the directory
    * @param name the name of the resource
    * @param file the file to write it into
    */
  def copyResourceToFile(name: String, file: File) {
    val in = getClass.getResourceAsStream(name)
    if (in == null) {
      throw new FileNotFoundException(s"Resource '$name'")
    }
    val out = new FileOutputStream(file)
    Iterator.continually(in.read).takeWhile(-1 != _).foreach(out.write)
    out.close()
  }

}
