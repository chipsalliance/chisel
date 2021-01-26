// SPDX-License-Identifier: Apache-2.0

package firrtl

import java.io.File

import firrtl.options.StageUtils

import scala.collection.Seq
import scala.sys.process.{stringSeqToProcess, BasicIO, ProcessLogger}

object FileUtils {

  /** Create a directory if it doesn't exist
    * @param directoryName a directory string with one or more levels
    * @return true if the directory exists or if it was successfully created
    */
  def makeDirectory(directoryName: String): Boolean = {
    val dirFile = new File(directoryName)
    if (dirFile.exists()) {
      dirFile.isDirectory
    } else {
      dirFile.mkdirs()
    }
  }

  /**
    * recursively delete all directories in a relative path
    * DO NOT DELETE absolute paths
    *
    * @param directoryPathName a directory hierarchy to delete
    */
  def deleteDirectoryHierarchy(directoryPathName: String): Boolean = {
    deleteDirectoryHierarchy(new File(directoryPathName))
  }

  /**
    * recursively delete all directories in a relative path
    * DO NOT DELETE absolute paths
    *
    * @param file: a directory hierarchy to delete
    */
  def deleteDirectoryHierarchy(file: File, atTop: Boolean = true): Boolean = {
    if (
      file.getPath.split("/").last.isEmpty ||
      file.getAbsolutePath == "/" ||
      file.getPath.startsWith("/")
    ) {
      StageUtils.dramaticError(s"delete directory ${file.getPath} will not delete absolute paths")
      false
    } else {
      val result = {
        if (file.isDirectory) {
          file.listFiles().forall(f => deleteDirectoryHierarchy(f)) && file.delete()
        } else {
          file.delete()
        }
      }
      result
    }
  }

  /** Indicate if an external command (executable) is available (from the current PATH).
    *
    * @param cmd the command/executable plus any arguments to the command as a Seq().
    * @return true if ```cmd <args>``` returns a 0 exit status.
    */
  def isCommandAvailable(cmd: Seq[String]): Boolean = {
    // Eat any output.
    val sb = new StringBuffer
    val ioToDevNull = BasicIO(withIn = false, ProcessLogger(line => sb.append(line)))

    try {
      cmd.run(ioToDevNull).exitValue() == 0
    } catch {
      case _: Throwable => false
    }
  }

  /** Indicate if an external command (executable) is available (from the current PATH).
    *
    * @param cmd the command/executable (without any arguments).
    * @return true if ```cmd``` returns a 0 exit status.
    */
  def isCommandAvailable(cmd: String): Boolean = {
    isCommandAvailable(Seq(cmd))
  }

  /** Flag indicating if vcs is available (for Verilog compilation and testing).
    * We used to use a bash command (`which ...`) to determine this, but this is problematic on Windows (issue #807).
    * Instead we try to run the executable itself (with innocuous arguments) and interpret any errors/exceptions
    *  as an indication that the executable is unavailable.
    */
  lazy val isVCSAvailable: Boolean =
    isCommandAvailable(Seq("vcs", "-platform")) || isCommandAvailable(Seq("vcs", "-full64", "-platform"))

  /** Read a text file and return it as a Seq of strings
    * Closes the file after read to avoid dangling file handles
    *
    * @param fileName The file to read
    */
  def getLines(fileName: String): Seq[String] = getLines(new File(fileName))

  /** Read a text file and return it as  a Seq of strings
    * Closes the file after read to avoid dangling file handles
    *
    * @param file a java File to be read
    */
  def getLines(file: File): Seq[String] = {
    val source = scala.io.Source.fromFile(file)
    val lines = source.getLines().toList
    source.close()
    lines
  }

  /** Read a text file and return it as  a single string
    * Closes the file after read to avoid dangling file handles
    *
    * @param fileName The file to read
    */
  def getText(fileName: String): String = getText(new File(fileName))

  /** Read a text file and return it as  a single string
    * Closes the file after read to avoid dangling file handles
    *
    * @param file a java File to be read
    */
  def getText(file: File): String = {
    val source = scala.io.Source.fromFile(file)
    val text = source.mkString
    source.close()
    text
  }

  /** Read text file and return it as  a Seq of strings
    * Closes the file after read to avoid dangling file handles
    * @note resourceName typically begins with a slash.
    *
    * @param resourceName a java File to be read
    */
  def getLinesResource(resourceName: String): Seq[String] = {
    val inputStream = getClass.getResourceAsStream(resourceName)
    // the .toList at the end is critical to force stream to be read.
    // Without it the lazy evaluation can cause failure in MultiThreadingSpec
    val text = scala.io.Source.fromInputStream(inputStream).getLines().toList
    inputStream.close()
    text
  }

  /** Read text file and return it as  a single string
    * Closes the file after read to avoid dangling file handles
    * @note resourceName typically begins with a slash.
    *
    * @param resourceName a java File to be read
    */
  def getTextResource(resourceName: String): String = {
    val inputStream = getClass.getResourceAsStream(resourceName)
    val text = scala.io.Source.fromInputStream(inputStream).mkString
    inputStream.close()
    text
  }
}
