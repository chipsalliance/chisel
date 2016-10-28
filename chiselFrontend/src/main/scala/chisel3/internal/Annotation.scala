// See LICENSE for license details.

package chisel3.internal

import net.jcazevedo.moultingyaml._

object MyYamlProtocol extends DefaultYamlProtocol {
  implicit val annotationFormat = yamlFormat3(Annotation)
}

case class Annotation(className: String, targetName: String, value: String) {

}
