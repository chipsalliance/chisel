FROM --platform=$BUILDPLATFORM ubuntu:20.04

ARG TARGETARCH

# Install Verilator
# Get Dependencies
RUN \
  DEBIAN_FRONTEND=noninteractive \
    apt-get update -y && \
    apt-get install -y \
      git perl python3 make autoconf clang flex bison ccache \
      libgoogle-perftools-dev numactl perl-doc \
      libfl2 \
      libfl-dev \
      zlibc zlib1g zlib1g-dev
# Build from Source
RUN \
  git clone -b v4.226 https://github.com/verilator/verilator /verilator-source && \
  cd /verilator-source && \
  autoconf && \
  ./configure && \
  make -j `nproc` &&\
  make install

# Install GraalVM
# This downloads all available GraalVM releases at once, mostly because $TARGETARCH values don't map exactly to the release URLs. Since we're optimizing for developer experience here and not image size, this is OK
# GraalVM release links can be found here: https://github.com/graalvm/graalvm-ce-builds/releases
ADD https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-22.3.1/graalvm-ce-java17-linux-aarch64-22.3.1.tar.gz /graalvm/tarballs/arm64.tar.gz
ADD https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-22.3.1/graalvm-ce-java17-linux-amd64-22.3.1.tar.gz /graalvm/tarballs/amd64.tar.gz
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
# Make sure sbt is ready to go
RUN sbt --version 
