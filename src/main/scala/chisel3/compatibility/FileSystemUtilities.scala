// See LICENSE for license details.

package chisel3.compatibility

import chisel3._

@deprecated("FileSystemUtilities doesn't exist in chisel3", "3.0.0")
trait FileSystemUtilities {
  def createOutputFile(name: String): java.io.FileWriter  = {
    new java.io.FileWriter(Driver.targetDir + "/" + name)
  }
}
