{
  description = "chisel";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
    circtSrc = {
      url = "git+https://github.com/llvm/circt?submodules=1&shallow=1";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, circtSrc }@inputs:
    let
      overlay = import ./overlay.nix circtSrc ;
    in
    flake-utils.lib.eachDefaultSystem
      (system:
        let
          pkgs = import nixpkgs { inherit system; overlays = [ overlay ]; };
          deps = with pkgs; [
            mill
            circt
            jextract
            llvm-lit
          ];
        in
        {
          legacyPackages = pkgs;
          devShell = pkgs.mkShell {
            buildInputs = deps;
            env = {
              CIRCT_INSTALL_PATH = pkgs.circt;
            };
          };
        }
      ) // { inherit inputs; overlays.default = overlay; };
}
