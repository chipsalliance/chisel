// See LICENSE for license details.

package firrtl.stage.transforms

import firrtl.{CircuitState, CustomTransformException, Transform}

class CatchCustomTransformExceptions(val underlying: Transform) extends Transform with WrappedTransform {

  override def execute(c: CircuitState): CircuitState = try {
    underlying.execute(c)
  } catch {
    case e: Exception if CatchCustomTransformExceptions.isCustomTransform(trueUnderlying) => throw CustomTransformException(e)
  }

}

object CatchCustomTransformExceptions {

  private[firrtl] def isCustomTransform(xform: Transform): Boolean = {
    def getTopPackage(pack: java.lang.Package): java.lang.Package =
      Package.getPackage(pack.getName.split('.').head)
    // We use the top package of the Driver to get the top firrtl package
    Option(xform.getClass.getPackage).map { p =>
      getTopPackage(p) != firrtl.Driver.getClass.getPackage
    }.getOrElse(true)
  }

  def apply(a: Transform): CatchCustomTransformExceptions = new CatchCustomTransformExceptions(a)

}
