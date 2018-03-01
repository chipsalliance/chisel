// See:
//  https://github.com/kciesielski/tags-demo
//  http://www.edgezero.de/2015/03/partitioning-your-test-code-with-scalatest-and-sbt/
//  http://www.michaelpollmeier.com/2015/08/26/tag-scalatest-suite
package tags;

import java.lang.annotation.*;

@org.scalatest.TagAnnotation
@Inherited
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.TYPE})
// If you want the same tag to control both test suites and individual tests,
//  the package and annotation name should match:
//      object TagRequiresBackend extends Tag("tags.RequiresBackend")
// in .../src/test/scala/tags/TagRequiresBackend.java
// To disable, use the following argument to `sbt test`:
//   -l tags.RequiresBackend
public @interface RequiresBackend {}
