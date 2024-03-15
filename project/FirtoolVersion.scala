// See LICENSE for license details.

object FirtoolVersion {
  def version: String = {
    val contents = os.read(os.pwd / "etc" / "circt.json")
    val read = upickle.default.read[Map[String, String]](contents)
    read("version").stripPrefix("firtool-")
  }
}
