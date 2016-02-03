// See LICENSE for details

package Chisel

@deprecated("FileSystemUtilities doesn't exist in chisel3", "3.0.0")
trait FileSystemUtilities {
  def createOutputFile(name: String) = {
    new java.io.FileWriter(Driver.targetDir + "/" + name)
  }
}
