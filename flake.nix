{
  description = "chisel";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }@inputs:
    let
      overlay = import ./overlay.nix;
    in
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs { inherit system; overlays = [ overlay ]; };
          deps = with pkgs; [
            jdk21
            mill
            circt
            jextract-21
            lit
            scala-cli
            llvm
            verilator
          ];
        in
        {
          legacyPackages = pkgs;
          devShell = pkgs.mkShell {
            buildInputs = deps;
            env = {
              CIRCT_INSTALL_PATH = pkgs.circt-all;
              JEXTRACT_INSTALL_PATH = pkgs.jextract-21;
            };
          };
        }
      ) // { inherit inputs; overlays.default = overlay; };
}
