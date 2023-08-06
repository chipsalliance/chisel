FROM --platform=$BUILDPLATFORM ubuntu:22.04

ARG TARGETARCH
ARG UBUNTU_VERSION_NAME=jammy

ENV LC_ALL C

# Update apt and install common tools
RUN DEBIAN_FRONTEND="noninteractive" \
      apt-get update && \
      apt-get install -y \
        ccache \
        curl \
        git \
        jq \
        ninja-build \
        rsync

# Install the correct version of CMake (the version from apt-get installs GCC, the presence of which sometimes confuses LLVM's build). Release URLS can be found at https://github.com/Kitware/CMake/releases
ARG CMAKE_VERSION=3.26.4
ADD https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-aarch64.sh /root/install-cmake-arm64.sh
ADD https://github.com/Kitware/CMake/releases/download/v$CMAKE_VERSION/cmake-$CMAKE_VERSION-linux-x86_64.sh /root/install-cmake-amd64.sh
RUN chmod +x /root/install-cmake-$TARGETARCH.sh && \
      /root/install-cmake-$TARGETARCH.sh --prefix=/usr/ --exclude-subdir --skip-license && \
      rm /root/install-cmake-*.sh

# Install clang
ARG CLANG_VERSION=16
ADD https://apt.llvm.org/llvm-snapshot.gpg.key /root/llvm-snapshot.gpg.key
RUN DEBIAN_FRONTEND="noninteractive" \
      apt-get install -y software-properties-common gnupg && \
      apt-key add /root/llvm-snapshot.gpg.key && \
      apt-add-repository "deb http://apt.llvm.org/$UBUNTU_VERSION_NAME/ llvm-toolchain-$UBUNTU_VERSION_NAME-$CLANG_VERSION main" && \
      apt update && \
      apt install -y \
        clang-$CLANG_VERSION \
        clangd-$CLANG_VERSION \
        lldb-$CLANG_VERSION \
        clang-tools-$CLANG_VERSION

# Use installed clang with CMake
COPY cmake-tools-kits.json /root/.local/share/CMakeTools/cmake-tools-kits.json

# Install Java (GraalVM)
# This downloads all relevant GraalVM architectures at once, mostly because $TARGETARCH values don't map exactly to the release URLs. Since we're optimizing for developer experience here and not image size, this is OK
# GraalVM release links can be found here: https://github.com/graalvm/graalvm-ce-builds/releases
ARG JAVA_VERSION=java11
ADD https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-22.3.1/graalvm-ce-$JAVA_VERSION-linux-aarch64-22.3.1.tar.gz /graalvm/tarballs/arm64.tar.gz
ADD https://github.com/graalvm/graalvm-ce-builds/releases/download/vm-22.3.1/graalvm-ce-$JAVA_VERSION-linux-amd64-22.3.1.tar.gz /graalvm/tarballs/amd64.tar.gz
RUN tar -xzf /graalvm/tarballs/$TARGETARCH.tar.gz -C /graalvm --strip-components=1 && \
    rm -rf /graalvm/tarballs
ENV JAVA_HOME=/graalvm \
    PATH=/graalvm/bin/:$PATH

# Install SBT
# Effectively `curl -fL https://github.com/VirtusLab/coursier-m1/releases/latest/download/cs-aarch64-pc-linux.gz | gzip -d > cs && chmod +x cs && ./cs setup` from https://www.scala-lang.org/download/
ADD https://github.com/VirtusLab/coursier-m1/releases/latest/download/cs-aarch64-pc-linux.gz /scala/coursier-install-arm64.gz
ADD https://github.com/coursier/coursier/releases/latest/download/cs-x86_64-pc-linux.gz /scala/coursier-install-amd64.gz
RUN cd /scala && \
    cat coursier-install-$TARGETARCH.gz | gzip -d > cs && \
    chmod +x cs && \
    yes | ./cs setup && \
    rm /scala/coursier-install-*.gz

# Install Swift Nightly Toolchain
ARG SWIFT_SNAPSHOT=DEVELOPMENT-SNAPSHOT-2023-07-23-a
ADD https://swift.org/keys/all-keys.asc /swift/all-keys.asc
ADD https://download.swift.org/development/ubuntu2204-aarch64/swift-$SWIFT_SNAPSHOT/swift-$SWIFT_SNAPSHOT-ubuntu22.04-aarch64.tar.gz /swift/toolchain-arm64.tar.gz
ADD https://download.swift.org/development/ubuntu2204/swift-$SWIFT_SNAPSHOT/swift-$SWIFT_SNAPSHOT-ubuntu22.04.tar.gz /swift/toolchain-amd64.tar.gz
RUN DEBIAN_FRONTEND="noninteractive" \
      apt-get install -y \
        binutils \
        git \
        gnupg2 \
        libc6-dev \
        libcurl4-openssl-dev \
        libedit2 \
        libgcc-9-dev \
        libpython3.8 \
        libsqlite3-0 \
        libstdc++-9-dev \
        libxml2-dev \
        libz3-dev \
        pkg-config \
        tzdata \
        unzip \
        zlib1g-dev && \
      cd /swift && \
      tar xzf toolchain-$TARGETARCH.tar.gz --strip-components=1 && \
      rm toolchain-*.tar.gz
ENV PATH="$PATH:/swift/usr/bin"

ENV EDITOR="code --wait" \
    GIT_EDITOR="code --wait"
