package chisel3.simulator

/** Changes that ChiselSim should make to Chisel command line options
  *
  * This follows a type class pattern with a low-priority default of identity.
  */
trait ChiselOptionsModifications extends (Array[String] => Array[String])

object ChiselOptionsModifications {

  /** Low-priority default of identity (no opotions modifications) */
  implicit def unmodified: ChiselOptionsModifications = identity(_)

}

/** Changes that ChiselSim should make to `firtool` command line options
  *
  * This follows a type class pattern with a low-priority default of identity.
  */
trait FirtoolOptionsModifications extends (Array[String] => Array[String])

object FirtoolOptionsModifications {

  /** Low-priority default of identity (no opotions modifications) */
  implicit def unmodified: FirtoolOptionsModifications = identity(_)

}
