// SPDX-License-Identifier: Apache-2.0

package firrtl

import firrtl.options.StageUtils

import scala.collection.Seq
import scala.sys.process.{stringSeqToProcess, BasicIO, ProcessLogger}

object FileUtils {

  /**
    * recursively delete all directories in a relative path
    * DO NOT DELETE absolute paths
    *
    * @param directoryPathName a directory hierarchy to delete
    */
  def deleteDirectoryHierarchy(directoryPathName: String): Boolean = {
    os.FilePath(directoryPathName) match {
      case path: os.Path =>
        StageUtils.dramaticError(s"delete directory $path will not delete absolute paths")
        false
      case rel: os.RelPath =>
        val path = os.pwd / rel
        os.exists(path) && { os.remove.all(path); true }
      case sub: os.SubPath =>
        val path = os.pwd / sub
        os.exists(path) && { os.remove.all(path); true }
    }
  }

  /** Read a text file and return it as a Seq of strings
    * Closes the file after read to avoid dangling file handles
    *
    * @param fileName The file to read
    */
  def getLines(fileName: String): Seq[String] = getLines(getPath(fileName))

  /** Read a text file and return it as  a Seq of strings
    * Closes the file after read to avoid dangling file handles
    * @param file an os.Path to be read
    */
  def getLines(file: os.Path): Seq[String] = os.read.lines(file)

  /** Read a text file and return it as  a Seq of strings
    * Closes the file after read to avoid dangling file handles
    *
    * @param file a java File to be read
    */
  @deprecated("Use os-lib instead, this function will be removed in FIRRTL 1.6", "FIRRTL 1.5")
  def getLines(file: java.io.File): Seq[String] = {
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
  def getText(fileName: String): String = getText(getPath(fileName))

  /** Read a text file and return it as  a single string
    * Closes the file after read to avoid dangling file handles
    *
    * @param file an os.Path to be read
    */
  def getText(file: os.Path): String = os.read(file)

  /** Read a text file and return it as  a single string
    * Closes the file after read to avoid dangling file handles
    *
    * @param file a java File to be read
    */
  @deprecated("Use os-lib instead, this function will be removed in FIRRTL 1.6", "FIRRTL 1.5")
  def getText(file: java.io.File): String = {
    val source = scala.io.Source.fromFile(file)
    val text = source.mkString
    source.close()
    text
  }

  /** Get os.Path from String
    * @param pathName an absolute or relative path string
    */
  def getPath(pathName: String): os.Path = os.FilePath(pathName) match {
    case path: os.Path    => path
    case sub:  os.SubPath => os.pwd / sub
    case rel:  os.RelPath => os.pwd / rel
  }
}
