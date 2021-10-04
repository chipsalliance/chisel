---
layout: docs
title:  "Test Coverage"
section: "chisel3"
---

# Test Coverage

## Test Coverage Setup

Chisel's sbt build instructions contain the requisite plug-in (sbt-scoverage) for generating test coverage information. Please see the [sbt-scoverage web page](https://github.com/scoverage/sbt-scoverage) for details on the plug-in.
The tests themselves are found in `src/test/scala`.

## Generating A Test Coverage Report

Use the following sequence of sbt commands to generate a test coverage report:
```
sbt clean coverage test
sbt coverageReport
```
The coverage reports should be found in `target/scala-x.yy/scoverage-report/{scoverage.xml,index.html}` where `x.yy` corresponds to the version of Scala used to compile Firrtl and the tests.
`scoverage.xml` is useful if you want to analyze the results programmatically.
`index.html` is designed for navigation with a web browser, allowing one to drill down to invidual statements covered (or not) by the tests.
