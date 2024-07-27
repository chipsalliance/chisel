package chisel3.simulator

import java.io.File
import chisel3.layer.Layer

/** Utilities for enabling and disabling Chisel layers */
object LayerControl {

  /** The type of all layer control variations */
  sealed trait Type {

    /** Return true if a file should be included in the build.  This will always
      * return true for any non-layer file.
      * @param file the file to check
      */
    final def filter(file: File): Boolean = file.getName match {
      case nonLayer if !nonLayer.startsWith("layers_") => true
      case layer                                       => shouldEnable(layer)
    }

    /** Return true if a layer should be included in the build.
      * @param layerFilename the filename of a layer
      */
    protected def shouldEnable(layerFilename: String): Boolean
  }

  /** Enable all layers */
  case object EnableAll extends Type {
    override protected def shouldEnable(layerFilename: String) = true
  }

  /** Disable all layers */
  case object DisableAll extends Type {
    override protected def shouldEnable(filename: String) = false
  }

  /** Enable only the specified layers
    *
    * Nested layers should use a `.` as a delimiter.
    *
    * @param layers a variadic list of layer names
    */
  case class Enable(layers: Layer*) extends Type {
    private val re = {
      val layersRe = layers.map(_.fullName.split("\\.").mkString("_")).mkString("|")
      s"^layers_\\w+_($layersRe)\\.sv$$".r
    }
    override protected def shouldEnable(filename: String) = re.matches(filename)
  }
}
