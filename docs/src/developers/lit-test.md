---
layout: docs
title:  "Lit Test"
section: "chisel3"
---

# Lit Test

We used llvm-lit and scala-cli to test CIRCT Converter. For how this tool works, see [lit - LLVM Integrated Tester](https://llvm.org/docs/CommandGuide/lit.html).

## Run tests

The first line of the test file indicates how the test will be run, mostly in the form of `scala-cli ... | FileCheck`.

To run all test cases, make sure the environment variables are defined first:

- `JAVA_HOME`: The directory where the Java runtime environment. The Panama framework requires exact Java 21.

```
mill -i lit[_].run
```

## Update test case

There is a lack of a convenient way to debug test cases. But the `println` debugging method always works.

You need to temporarily modify `tests.sc` to make `lit` output more detailed (stdout, stderr).

```diff
-os.proc("lit", litConfig().path)
+os.proc("lit", litConfig().path, "-a")
```

If the output of `FileCheck` confuses you, you will also need to temporarily remove the `| FileCheck ...` from the test case file header.

Then re-run `mill -i lit[_].run` and you will see the output of the failed test cases. Check that they are correct and then update the test case.

#### Debugging

When interacting with CIRCT, it is easy to get a JVM crash, which is usually due to an unexpected error inside CIRCT (e.g. an assertion failure). Ensuring that CIRCT has been compiled with debug information (CMake option `-DCMAKE_BUILD_TYPE=RelWithDebInfo`) and then checking the call stack of the crash will help you to localize the bug.
