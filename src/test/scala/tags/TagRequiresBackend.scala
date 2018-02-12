package tags

import org.scalatest.Tag

// If you want the same tag to control both test suites and individual tests, the tag name should match:
//  public @interface RequiresBackend {}
// from .../src/test/java/tags/RequiresBackend.java
// with the addition of the package name - "tags".
// To disable, use the following argument to `sbt test`:
//   -l tags.RequiresBackend
object TagRequiresBackend extends Tag("tags.RequiresBackend")
