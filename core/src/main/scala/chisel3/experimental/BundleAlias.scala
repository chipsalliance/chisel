package chisel3.experimental

case class BundleAlias private[chisel3] (info: SourceInfo, id: String)

object BundleAlias {
  def apply(id: String)(implicit info: SourceInfo): BundleAlias = BundleAlias(info, id)
}
