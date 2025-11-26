---
description: How to install and run Chisel
toc_max_heading_level: 4
---

# Installation

Chisel is a [Scala](https://www.scala-lang.org/) library and compiler plugin.

## Quickstart with Scala CLI

The easiest way to use Chisel is to:

1. First, [install Scala CLI](https://scala-cli.virtuslab.org/install).

2. Then, download the Chisel Scala CLI example.

This is easiest on the command-line (works on Linux, MacOS, and Windows Subsystem for Linux (WSL)):
```bash
curl -O -L https://github.com/chipsalliance/chisel/releases/latest/download/chisel-example.scala
```

Alternatively you can download the example directly from [this link](https://github.com/chipsalliance/chisel/releases/latest/download/chisel-example.scala).

3. Finally, use Scala CLI to compile and run the example:
```bash
scala-cli chisel-example.scala
```

The Chisel Scala CLI example is a simple, single-file example of Chisel that emits Verilog to the screen.

:::tip

While more complex projects often use a build tool like SBT or Mill as described below,
we still highly recommend using Scala CLI with the example linked above for experimentation and
writing small snippets to share with others.

:::

## Dependencies

As described above, Scala CLI is a great "batteries included" way to use Chisel.
It will automatically download and manage all dependencies of Chisel _including a Java Development Kit (JDK)_.
More complex projects will require the user to install a JDK and a build tool.

:::info

Note that each of these dependencies are projects with their own installation instructions.
Please treat the commands below as suggestions and not directives.

:::

### Java Development Kit (JDK)

Scala runs on the Java Virtual Machine (JVM), so it is necessary to install a JDK to use Chisel.
Chisel works on any version of Java version 8 or newer; however, we recommend using an LTS release version 17 or newer.
Note that Scala CLI requires Java 17 or newer so unless your system installation of Java is at least version 17, Scala CLI will download Java 17 for its own use.

You can install any distribution of the JDK you prefer.
Eclipse Adoptium Temurin is a good option with support for all platforms: https://adoptium.net/installation

#### Ubuntu

Note that Temurin is not part of the default apt repositories, so you will need to add the Eclipse Adoptium repository.

Taken from the official [Temurin docs](https://adoptium.net/installation/linux/).
Please note that installation on most systems will require superuser priviledges (`sudo`).
You may need to prefix these commands with `sudo` including the `tee` commands following any pipes (`|`).

```sh
# Ensure the necessary packages are present:
apt install -y wget gpg apt-transport-https

# Download the Eclipse Adoptium GPG key:
wget -qO - https://packages.adoptium.net/artifactory/api/gpg/key/public | gpg --dearmor | tee /etc/apt/trusted.gpg.d/adoptium.gpg > /dev/null

# Configure the Eclipse Adoptium apt repository
echo "deb https://packages.adoptium.net/artifactory/deb $(awk -F= '/^VERSION_CODENAME/{print$2}' /etc/os-release) main" | tee /etc/apt/sources.list.d/adoptium.list

# Update the apt packages
apt update

# Install
apt install temurin-17-jdk
```

#### MacOS

Using [MacPorts](https://www.macports.org):
```sh
sudo port install openjdk17-temurin
```

Using [Homebrew](https://brew.sh):
```sh
brew install temurin@17
```

#### Windows

Using [Scoop](https://scoop.sh):
```sh
scoop install temurin17-jdk
```

Windows users may also prefer using an [installer](https://adoptium.net/temurin/releases/?version=17&os=windows&arch=x86&package=jdk).

#### Java Versions

Different JVM versions require a [minimum Scala version](https://docs.scala-lang.org/overviews/jdk-compatibility/overview.html).
Similarly, different Scala versions have a minimum Chisel version.

Scala/Java compatibility table (maximum version supported):

| Chisel | Scala   | Java
|--------|---------|------
| 3.5.6  | 2.13.10 | 17
| 3.6.1  | 2.13.14 | 22
| 5.x    | 2.13.14 | 22
| 6.x    | 2.13.16 | 23
| 7.x    | 2.13.18 | 26

Note that, by default, `brew` on Mac installs the newest version of Java which often is not LTS and not yet supported by the maximum version of Scala supported by Chisel.
Stick to LTS and you should have no issues (e.g. `brew install openjdk@17`)

### Build Tools

Scala CLI is only intended for projects made up of a single to a handful of files.
Larger projects need a build tool.

#### Mill

[Mill](https://mill-build.com) is a modern Scala build tool with simple syntax and a better command-line experience than SBT.
We recommend Chisel users use Mill.

For detailed instructions, please see the [Mill documentation](https://mill-build.com/mill/Intro_to_Mill.html).

##### Linux and MacOS

The easiest way to use Mill is with the Mill Wrapper Script `millw`:

```sh
curl -L https://raw.githubusercontent.com/lefou/millw/0.4.11/millw > mill && chmod +x mill
```

You can then move this script to a global install location:

```sh
sudo mv mill /usr/local/bin/
```

##### Windows

Using [Scoop](https://scoop.sh):
```sh
scoop install mill
```

<!-- TODO flesh this out -->
Download `millw.bat`: https://raw.githubusercontent.com/lefou/millw/0.4.11/millw.bat.

#### SBT

[SBT](https://www.scala-sbt.org) is the more traditional Scala build tool with many resources and examples.
It is most productively used with its REPL rather than on the command line.

##### Linux

The easiest way to install SBT is manually from the release tarball.
Note that MacOS and Windows users can also do a manual install.

```sh
curl -s -L https://github.com/sbt/sbt/releases/download/v1.9.7/sbt-1.9.7.tgz | tar xvz
```

Then copy the `sbt` bootstrap script into a global install location.

```sh
sudo mv sbt/bin/sbt /usr/local/bin/
```

##### MacOS

Using [MacPorts](https://www.macports.org):
```sh
sudo port install sbt
```

Using [Homebrew](https://brew.sh/):

```sh
brew install sbt
```

:::warning

Note that `brew` installs the latest version of Java as a dependency of `sbt` even though that version of Java is not actually supported by SBT.
Users are advised to remove the `brew` version of Java with the following command:

```sh
brew uninstall --ignore-dependencies java
```

:::

##### Windows

Using [Scoop](https://scoop.sh):
```sh
scoop install sbt
```

### Firtool

Beginning with version 6.0, Chisel manages the version of firtool on most systems.
However, some systems (e.g. NixOS or older Linux distributions like CentOS 6) may need to build firtool from source.
If you need to build firtool from source, please see the [Github repository](https://github.com/llvm/circt).

To override the Chisel-managed version of firtool, set environment variable `CHISEL_FIRTOOL_PATH` to point to the directory containing your firtool binary.

To see what version of firtool is recommended for a given version of Chisel, see [Versioning](appendix/versioning#firtool-version).
You can also query this information programmatically in Scala via [`chisel3.BuildInfo.firtoolVersion`](https://javadoc.io/doc/org.chipsalliance/chisel_2.13/latest/chisel3/BuildInfo$.html).
For example, you can use Scala CLI to compile a tiny program on the command-line to print out this value:

```bash
scala-cli -S 2.13 -e 'println(chisel3.BuildInfo.firtoolVersion)' --dep org.chipsalliance::chisel:7.2.0
```

### Verilog Simulation

#### Verilator

[Verilator](https://www.veripool.org/verilator/) is a high-performance, open-source Verilog simulator.
It is not a simulator in the traditional sense, but rather it works by transpiling your Verilog code to C++ which you then compile into a binary.
This results in Verilator itself having additional requirements, like a C++ compiler supporting at least C++14, and Make.

Please see [Verilator's install page](https://veripool.org/guide/latest/install.html) for more detailed instructions.

##### Linux

Most Linux package managers include Verilator, for example:

```sh
apt install -y verilator
```

Note that the default version is likely to be old, especially on older Linux distributions.
Users are encouraged to build Verilator from source, please see [the instructions on the Verilator website](https://veripool.org/guide/latest/install.html#git-quick-install).

##### MacOS

Using [MacPorts](https://www.macports.org):
```sh
sudo port install verilator
```

Using [Homebrew](https://brew.sh/):
```sh
brew install verilator
```

##### Windows

As mentioned above, Verilator is not a "single-executable" solution; it requires a C++ compiler and also uses quite a bit of Perl scripts.
We recommend using the Yosys OSS CAD Suite build for Windows, see [the install instructions](https://github.com/YosysHQ/oss-cad-suite-build?tab=readme-ov-file#installation).
Note that the Yosys OSS CAD Suite requires [MinGW (Minimalist GNU for Windows)](https://sourceforge.net/projects/mingw/).

Any Windows users who would like to help improve this usability of Chisel on Windows are encouraged to reach out.
See how to get in contact on the [Chisel community page](/community).
