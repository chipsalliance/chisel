package chisel3.simulator

import java.io.File
import chisel3.RawModule
import chisel3.layer.{ABI, Layer}
import chisel3.simulator.ElaboratedModule
import chisel3.stage.DesignAnnotation
import svsim.CommonCompilationSettings.VerilogPreprocessorDefine

/** Utilities for enabling and disabling Chisel layers */
object LayerControl {

  /** The type of all layer control variations */
  sealed trait Type {

    /** Return the layers that should be enabled in a circuit.  The layers must exist in the circuit.
      *
      * @param allLayers all layers that are defined in a circuit
      * @return the layers that should be enabled
      * @throws IllegalArgumentException if the requested layers are not in `allLayers`
      */
    protected def getLayerSubset(allLayers: Seq[Layer]): Seq[Layer]

    /** Return the preprocessor defines that should be set to enable the layers of
      * this `LayerControl.Type`.
      *
      * @param module a Chisel module
      * @param allLayers all the layers that are allow
      * @return preprocessor defines to control the enabling of these layers
      */
    final def preprocessorDefines(
      module:    RawModule,
      allLayers: Seq[Layer]
    ): Seq[VerilogPreprocessorDefine] = getLayerSubset(allLayers).flatMap {
      case layer =>
        layer.config.abi match {
          case abi: chisel3.layer.ABI.PreprocessorDefine.type =>
            Some(VerilogPreprocessorDefine(abi.toMacroIdentifier(layer, module.circuitName)))
          case _ => None
        }
    }

    /** Return the preprocessor defines that should be set to enable the layers of
      * this `LayerControl.Type`.
      *
      * This requires passing an elaborated module in order to know what layers
      * exist in the design.
      *
      * @param module an elaborated Chisel module
      * @return preprocessor defines to control the enabling of these layers
      */
    final def preprocessorDefines(
      module: ElaboratedModule[_ <: RawModule]
    ): Seq[VerilogPreprocessorDefine] = preprocessorDefines(module.wrapped, module.layers)

    /** Return a partial function that will return true if a file should be included
      * in the build to enable a layer.  This partial function is not defined if
      * the file is not a layer file.
      *
      * @param module a Chisel module
      * @param allLayers all the layers that can be enabled
      * @return a partial function to test if layer files should be included
      */
    final def shouldIncludeFile(
      module:    RawModule,
      allLayers: Seq[Layer]
    ): PartialFunction[File, Boolean] = {
      val layerFilenames: Seq[String] = getLayerSubset(allLayers).flatMap {
        case layer =>
          layer.config.abi match {
            case abi: chisel3.layer.ABI.FileInclude.type =>
              Some(abi.toFilename(layer, module.circuitName))
            case _ => None
          }
      }

      {
        case a if a.getName().startsWith("layers-") =>
          layerFilenames.mkString("|").r.matches(a.getName())
      }
    }

    /** Return a partial function that will return true if a file should be included
      * in the build to enable a layer.  This partial function is not defined if
      * the file is not a layer file.
      *
      * @param module an elaborated Chisel module
      * @return a partial function to test if layer files should be included
      */
    final def shouldIncludeFile(
      module: ElaboratedModule[_ <: RawModule]
    ): PartialFunction[File, Boolean] = shouldIncludeFile(module.wrapped, module.layers)

  }

  /** Enable all layers */
  final case object EnableAll extends Type {

    override protected def getLayerSubset(layers: Seq[Layer]): Seq[Layer] = layers
  }

  /** Enable only the specified layers
    *
    * Nested layers should use a `.` as a delimiter.
    *
    * @param layers a variadic list of layer names
    */
  final case class Enable(layers: Layer*) extends Type {
    private val _shouldEnable: String => Boolean = {
      layers match {
        case Nil => _ => false
        case _ =>
          val layersRe = layers.map(_.fullName.split("\\.").mkString("-")).mkString("|")
          val re = s"^layers-\\w+-($layersRe)\\.sv$$".r
          re.matches(_)
      }
    }

    override protected def getLayerSubset(allLayers: Seq[Layer]): Seq[Layer] = {
      val layerSet = allLayers.toSet
      layers.foreach { layer =>
        require(
          layerSet.contains(layer),
          s"""cannot enable layer '${layer.fullName}' as it is not one of the defined layers: ${allLayers.map(
            _.fullName
          )}"""
        )
      }
      layers
    }
  }

  /** Disables all layers.  This is the same as `Enable()`. */
  val DisableAll = Enable()

}
