FROM --platform=$BUILDPLATFORM ubuntu:22.04

ARG TARGETARCH

# Initialize System
RUN \
  DEBIAN_FRONTEND=noninteractive \
    apt-get update -y && \
    apt-get install -y \
      locales
RUN echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && locale-gen

# Install Verilator
# Get Dependencies
RUN \
  DEBIAN_FRONTEND=noninteractive \
    apt-get update -y && \
    apt-get install -y \
      git perl python3 make autoconf clang flex bison ccache \
      libgoogle-perftools-dev numactl perl-doc \
      libfl2 \
      libfl-dev
# Build from Source
RUN \
  git clone -b v4.226 https://github.com/verilator/verilator /verilator-source && \
  cd /verilator-source && \
  autoconf && \
  ./configure && \
  make -j `nproc` &&\
  make install

# Install GraalVM
# This downloads all relevant GraalVM architectures at once, mostly because $TARGETARCH values don't map exactly to the release URLs. Since we're optimizing for developer experience here and not image size, this is OK
# GraalVM release links can be found here: https://github.com/graalvm/graalvm-ce-builds/releases
ENV JAVA_VERSION=java11
ADD https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-22.3.1/graalvm-ce-$JAVA_VERSION-linux-aarch64-22.3.1.tar.gz /graalvm/tarballs/arm64.tar.gz
ADD https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-22.3.1/graalvm-ce-$JAVA_VERSION-linux-amd64-22.3.1.tar.gz /graalvm/tarballs/amd64.tar.gz
RUN tar -xzf /graalvm/tarballs/$TARGETARCH.tar.gz -C /graalvm --strip-components=1
ENV JAVA_HOME=/graalvm
ENV PATH=$JAVA_HOME/bin/:$PATH

# Install SBT via CS
# As above, grab the scripts for both platforms and choose the correct one using $TARGETARCH
# Release links can be found here: https://www.scala-lang.org/download/
ADD https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz /scala/cs-amd64.gz
ADD https://github.com/VirtusLab/coursier-m1/releases/latest/download/cs-aarch64-pc-linux.gz /scala/cs-arm64.gz
RUN gzip -d /scala/cs-$TARGETARCH.gz && \
    ln -s /scala/cs-$TARGETARCH /scala/cs && \
    chmod +x /scala/cs && \
    /scala/cs setup -y --dir /scala/bin
ENV PATH=/scala/bin:$PATH
# Preheat SBT
RUN sbt --version 

# Install firtool
RUN \
  DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
      cmake \
      ninja-build \
      curl

# Install recent git (required for `ls-tree --object-only`)
RUN \
  DEBIAN_FRONTEND=noninteractive \
    apt-get install -y software-properties-common && \
    apt-add-repository ppa:git-core/ppa && \
    apt-get update -y && \
    apt-get install -y git
RUN git clone https://github.com/llvm/circt /circt/source
# Grab the llvm source from GitHub directly to avoid having to clone the entire llvm-project repo
RUN cd /circt/source && \
    curl -L https://github.com/llvm/llvm-project/tarball/$(git ls-tree --object-only HEAD -- llvm) | tar xz --strip-components=1 -C llvm
# Install firtool
RUN cd /circt/source && \
    cmake \
      -S llvm/llvm \
      -B ../build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DLLVM_ENABLE_PROJECTS=mlir \
      -DLLVM_EXTERNAL_PROJECTS=circt \
      -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=. && \
    cmake \
      --build ../build \
      --target install-firtool
    
# Disable Chisel argument extensions, which cause issues when running tests from the VSCode UI
ENV CHISEL_ARGUMENT_EXTENSIONS=DISABLE

# Use VSCode for writing commit messages
ENV GIT_EDITOR="code --wait"
