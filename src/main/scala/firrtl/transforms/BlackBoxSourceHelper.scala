// See LICENSE for license details.

package firrtl.transforms

import java.io.{File, FileNotFoundException, FileOutputStream, PrintWriter}

import firrtl._
import firrtl.Utils.throwInternalError
import firrtl.annotations._

import scala.collection.immutable.ListSet

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
  def collectAnnos(annos: Seq[Annotation]): (ListSet[BlackBoxHelperAnno], File) =
    annos.foldLeft((ListSet.empty[BlackBoxHelperAnno], DefaultTargetDir)) {
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

    val resourceFiles: ListSet[File] = annos.collect {
      case BlackBoxResourceAnno(_, resourceId) =>
        BlackBoxSourceHelper.writeResourceToDirectory(resourceId, targetDir)
    }

    val inlineFiles: ListSet[File] = annos.collect {
      case BlackBoxInlineAnno(_, name, text) =>
        val outFile = new File(targetDir, name)
        (text, outFile)
    }.map { case (text, file) =>
      BlackBoxSourceHelper.writeTextToFile(text, file)
      file
    }

    BlackBoxSourceHelper.writeFileList(resourceFiles ++ inlineFiles, targetDir)

    state
  }
}

object BlackBoxSourceHelper {
  /**
    * finds the named resource and writes into the directory
    * @param name the name of the resource
    * @param dir the directory in which to write the file
    * @return the closed File object
    */
  def writeResourceToDirectory(name: String, dir: File): File = {
    val fileName = name.split("/").last
    val outFile = new File(dir, fileName)
    copyResourceToFile(name, outFile)
    outFile
  }

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

  val fileListName = "firrtl_black_box_resource_files.f"

  def writeFileList(files: ListSet[File], targetDir: File) {
    if (files.nonEmpty) {
      // We need the canonical path here, so verilator will create a path to the file that works from the targetDir,
      //  and, so we can compare the list of files automatically included, with an explicit list provided by the client
      //  and reject duplicates.
      // If the path isn't canonical, when make tries to determine dependencies based on the *__ver.d file, we end up with errors like:
      //  make[1]: *** No rule to make target `test_run_dir/examples.AccumBlackBox_PeekPokeTest_Verilator345491158/AccumBlackBox.v', needed by `.../chisel-testers/test_run_dir/examples.AccumBlackBox_PeekPokeTest_Verilator345491158/VAccumBlackBoxWrapper.h'.  Stop.
      //  or we end up including the same file multiple times.
      writeTextToFile(files.map(_.getCanonicalPath).mkString("\n"), new File(targetDir, fileListName))
    }
  }

  def writeTextToFile(text: String, file: File) {
    val out = new PrintWriter(file)
    out.write(text)
    out.close()
  }
}
