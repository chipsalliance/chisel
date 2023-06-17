final: prev:
{
  mill = (prev.mill.overrideAttrs (oldAttrs: rec {
    version = "0.11.2";
    src = prev.fetchurl {
      url = "https://github.com/com-lihaoyi/mill/releases/download/${version}/${version}-assembly";
      hash = "sha256-7RYMj/vfyzBQhZUpWzEaZYN27ZhYCRyKhQUhlH8tE0U=";
    };
  })).override {
    jre = final.openjdk20; 
  };
  espresso = final.callPackage ./nix/espresso.nix { };
  circt = prev.circt.overrideAttrs (old: {
    version = "firtool-1.54.0";
    cmakeFlags = [
      "-DBUILD_SHARED_LIBS=ON"
      "-DLLVM_ENABLE_BINDINGS=OFF"
      "-DLLVM_ENABLE_OCAMLDOC=OFF"
      "-DLLVM_BUILD_EXAMPLES=OFF"
      "-DLLVM_OPTIMIZED_TABLEGEN=ON"
      "-DLLVM_ENABLE_PROJECTS=mlir"
      "-DLLVM_EXTERNAL_PROJECTS=circt"
      "-DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=.."
      "-DCIRCT_LLHD_SIM_ENABLED=OFF"
    ];
    src = final.fetchFromGitHub {
      owner = "llvm";
      repo = "circt";
      rev = "firtool-1.54.0";
      hash = "sha256-jHDQl6UJTyNGZ4PUTEiZCIN/RSRbBxlaVutkwrWbK9M=";
      fetchSubmodules = true;
    };
    installPhase = ''
      runHook preInstall
      mkdir -p $out
      CMAKE_INSTALL_PREFIX=$out cmake --build . --target install --config Release
      runHook postInstall
    '';
  });

}
