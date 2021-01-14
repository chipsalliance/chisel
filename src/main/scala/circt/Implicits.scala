package circt

object Implicits {

  implicit class BooleanImplicits(a: Boolean) {

    /** Construct an Option from a Boolean. */
    def option[A](b: => A): Option[A] =
      if (a)
        Some(b)
      else
        None
  }

}
