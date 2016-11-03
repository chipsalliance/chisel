// See LICENSE for license details.

package ipxact

import org.scalatest.{FlatSpec, Matchers}

import org.accellera.spirit.v1685_2009._
import javax.xml.bind._

/* IPXACT Tests */
class IPXACTSpec extends FlatSpec with Matchers {
  val vlnv = new LibraryRefType
  vlnv.setLibrary("ucb-bar")
  vlnv.setName("IPXACTSpec")
  vlnv.setVendor("edu.berkeley.cs")
  vlnv.setVersion("1.0")

  val component = new ComponentInstance
  component.setComponentRef((vlnv))

  val componentInstances = new ComponentInstances
  var componentList = componentInstances.getComponentInstance
  componentList.add(component)
  val design = new Design
  design.setLibrary(vlnv.getLibrary)
  design.setName(vlnv.getName)
  design.setVendor(vlnv.getVendor)
  design.setVersion(vlnv.getVersion)
  design.setComponentInstances(componentInstances)

  // TODO: Set various flavors of component interconnections.

  behavior of "IPXACTSpec"

  val context = JAXBContext.newInstance(classOf[Design])
  it should "generate boiler plate xml" in {
    val marshaller = context.createMarshaller()
    marshaller.setProperty(Marshaller.JAXB_FORMATTED_OUTPUT, true)
    marshaller.marshal(design, System.out)
  }
}
