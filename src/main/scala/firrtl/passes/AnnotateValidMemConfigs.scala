package firrtl.passes

import firrtl.ir._
import firrtl._
import net.jcazevedo.moultingyaml._
import net.jcazevedo.moultingyaml.DefaultYamlProtocol._
import AnalysisUtils._
import scala.collection.mutable.ArrayBuffer

object CustomYAMLProtocol extends DefaultYamlProtocol {
  // bottom depends on top
  implicit val dr = yamlFormat4(DimensionRules)
  implicit val md = yamlFormat2(MemDimension)
  implicit val sr = yamlFormat4(SRAMRules)
  implicit val sc = yamlFormat11(SRAMCompiler)
}

case class DimensionRules(
  min: Int,
  // step size
  inc: Int,
  max: Int,
  // these values should not be used, regardless of min,inc,max
  illegal: Option[List[Int]]
){
  def getValid = {
    val range = (min to max by inc).toList
    range.filterNot(illegal.getOrElse(List[Int]()).toSet)
  }
}

case class MemDimension(
  rules: Option[DimensionRules],
  set: Option[List[Int]]
){
  require (
    if(rules == None) set != None else set == None, 
    "Should specify either rules or a list of valid options, but not both"
  )
  def getValid = set.getOrElse(rules.get.getValid)
}

case class SRAMConfig(
  ymux: String,
  ybank: String,
  width: String,
  depth: String
){
  def serialize(pattern: String): String = {
    val fieldMap = getClass.getDeclaredFields.map{f => 
      f.setAccessible(true)
      f.getName -> f.get(this)
    }.toMap

    val fieldDelimiter = """\[.*?\]""".r
    val configOptions = fieldDelimiter.findAllIn(pattern).toList

    configOptions.foldLeft(pattern)((b,a) => {
      // Expects the contents of [] are valid configuration fields (otherwise key match error)
      val fieldVal = {
        try fieldMap(a.substring(1,a.length-1)) 
        catch { case e: Exception => Error("**SRAM config field incorrect**") }
      }
      b.replace(a,fieldVal.toString)
    })
  }
}

// Ex: https://www.ece.cmu.edu/~ece548/hw/hw5/meml80.pdf
case class SRAMRules(
  // column mux parameter (for adjusting aspect ratio)
  ymux: (Int,String),
  // vertical segmentation (banking -- tradeoff performance / area)
  ybank: (Int,String),
  width: MemDimension,
  depth: MemDimension
){
  def getValidWidths = width.getValid
  def getValidDepths = depth.getValid
  def getValidConfig(m: DefMemory): Option[SRAMConfig] = {
    val width = bitWidth(m.dataType)
    val depth = m.depth
    if (getValidWidths.contains(width) && getValidDepths.contains(depth)) 
      Some(SRAMConfig(ymux = ymux._2, ybank = ybank._2, width = width.toString, depth = depth.toString))
    else
      None
  }

} 

// vendor-specific compilers
case class SRAMCompiler(
    vendor: String,
    node: String,
    // i.e. RF, SRAM, etc.
    memType: String,
    portType: String,
    useWmask: Boolean,
    // area of individual bitcells (um2) to determine periphery overhead
    bitCellArea: Option[Double],
    // rules for valid SRAM flavors
    rules: Seq[SRAMRules],
    // path to executable 
    path: Option[String],
    // (output) config file path
    configFile: Option[String],
    // config pattern
    configPattern: Option[String],
    // read documentation for details 
    defaultArgs: Option[String]
){
  require(portType == "RW" || portType == "R,W", "Memory must be single port RW or dual port R,W")
  require(
    (configFile != None && configPattern != None) || configFile == None, 
    "Config pattern must be provided with config file"
  ) 
  def ymuxVals = rules.map(_.ymux._1).sortWith(_ < _)
  def ybankVals = rules.map(_.ybank._1).sortWith(_ > _)
  // optimize search for better FoM (area,power,clk); ymux has more effect
  def defaultSearchOrdering = for (x <- ymuxVals; y <- ybankVals) yield {
    rules.find(r => r.ymux._1 == x && r.ybank._1 == y).get
  }

  private val configOutputBuffer = new java.io.CharArrayWriter

  def append(m: DefMemory) : Option[SRAMConfig] = {
    val validCombos = ArrayBuffer[SRAMConfig]()
    defaultSearchOrdering foreach { r =>
      val config = r.getValidConfig(m)
      if (config != None) validCombos += config.get
    }
    // non empty if successfully found compiler option that supports depth/width
    if (validCombos.nonEmpty){ 
      if (configPattern != None) 
        configOutputBuffer.append(validCombos.head.serialize(configPattern.get))
      Some(validCombos.head)
    }
    else None
  }

  // # of mems with given width, depth to make up the memory you want
  private case class MemInfo(num: Int, width: Int, depth: Int)

  // split memory until width, depth achievable via given memory compiler
  private def getInRange(m: MemInfo): Seq[MemInfo] = {
    val validXRange = ArrayBuffer[SRAMRules]()
    val validYRange = ArrayBuffer[SRAMRules]()
    defaultSearchOrdering foreach { r => 
      if (m.width < r.getValidWidths.max) validXRange += r
      if (m.depth < r.getValidDepths.max) validYRange += r
    }
    if (validXRange.isEmpty && validYRange.isEmpty)
      getInRange(MemInfo(4*m.num,m.width/2,m.depth/2))
    else if (validXRange.isEmpty && validYRange.nonEmpty)
      getInRange(MemInfo(2*m.num,m.width/2,m.depth)) 
    else if (validXRange.nonEmpty && validYRange.isEmpty)
      getInRange(MemInfo(2*m.num,m.width,m.depth/2)) 
    else if (validXRange.union(validYRange).nonEmpty)
      Seq(m)
    else 
      getInRange(MemInfo(2*m.num,m.width,m.depth/2)) ++ getInRange(MemInfo(2*m.num,m.width/2,m.depth))    
  }

}  

class YamlFileReader(file: String){
  import CustomYAMLProtocol._
  def parse: Seq[YamlValue] = {
    val yamlString = scala.io.Source.fromFile(file).getLines.mkString("\n")
    yamlString.parseYamls
  }
}

class AnnotateValidMemConfigs(reader: Option[YamlFileReader]) extends Pass {

  def name = "Annotate memories with valid split depths, widths, #\'s"

  def run(c: Circuit) = {

    def annotateModMems(m: Module) = {
      
      def updateStmts(s: Statement): Statement = s match {
        
        case m: DefMemory if containsInfo(m.info,"useMacro") => m   
        case b: Block => Block(b.stmts map updateStmts)
        case s => s

      }
      m.copy(body=updateStmts(m.body))
    } 

    val updatedMods = c.modules map {
      case m: Module => annotateModMems(m)
      case m: ExtModule => m
    }
    c.copy(modules = updatedMods)

  }  

}