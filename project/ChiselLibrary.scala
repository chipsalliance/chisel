import sbt._

// A class to represent chisel dependent libraries.
case class ChiselLibrary(organization: String, moduleName: String, defaultVersion: String, configuration: Option[String] = None) {
  val version = sys.props.getOrElse(moduleName + "Version", defaultVersion)

  def toSbtModuleId = {
    configuration match {
      case None => organization %% moduleName % version
      case Some(config: String) => organization %% moduleName % version % config
    }
  }
}
