package chisel3.simulator.stimulus

/** Types and utility functions related to the [[stimulus]] package. */
object Stimulus {

  /** The type of stimulus.  This takes a type and applies some stimulus to it.  Nothing is returned. */
  type Type[A] = (A) => Unit

}
