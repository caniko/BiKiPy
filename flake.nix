{
  description = "Bikipy — kinematic data analysis workspace";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    crane.url = "github:ipetkov/crane";

    rs-harbor = {
      url = "github:caniko/rs-harbor/e2778ff3beca1bd4c1f5183313251d1fb5b46dd6";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.crane.follows = "crane";
    };

    flake-utils.url = "github:numtide/flake-utils";

    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      crane,
      rs-harbor,
      flake-utils,
      advisory-db,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ (import rs-harbor.inputs.rust-overlay) ];
        };

        inherit (pkgs) lib;

        toolchain = rs-harbor.lib.mkToolchain { inherit pkgs; toolchainProfile = "nightly"; };
        craneLib = toolchain.craneLib;
        src = craneLib.cleanCargoSource ./.;

        commonArgs = {
          inherit src;
          strictDeps = true;

          buildInputs = [
          ]
          ++ lib.optionals pkgs.stdenv.isDarwin [
            pkgs.libiconv
          ];
        };

        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        individualCrateArgs = commonArgs // {
          inherit cargoArtifacts;
          inherit (craneLib.crateNameFromCargoToml { inherit src; }) version;
          doCheck = false;
        };

        fileSetForCrate =
          crate:
          lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./Cargo.toml
              ./Cargo.lock
              (craneLib.fileset.commonCargoSources ./crates/bikipy-core)
              (craneLib.fileset.commonCargoSources ./crates/bikipy-math)
              (craneLib.fileset.commonCargoSources ./crates/bikipy-perimeter)
              (craneLib.fileset.commonCargoSources ./crates/bikipy-reader)
              (craneLib.fileset.commonCargoSources ./crates/bikipy-feature)
              (craneLib.fileset.commonCargoSources ./crates/bikipy-behaviour)
              (craneLib.fileset.commonCargoSources ./crates/bikipy-ingress)
              (craneLib.fileset.commonCargoSources ./crates/bikipy-workspace-hack)
              (craneLib.fileset.commonCargoSources crate)
            ];
          };

        # Build the CLI as the top-level binary derivation.
        bikipy-cli = craneLib.buildPackage (
          individualCrateArgs
          // {
            pname = "bikipy-cli";
            cargoExtraArgs = "-p bikipy-cli";
            src = fileSetForCrate ./crates/bikipy-cli;
          }
        );

        # Python inspection package — matplotlib visualizations driven by Rust output.
        bikipy-inspect = pkgs.python312.pkgs.buildPythonApplication {
          pname = "bikipy-inspect";
          version = "0.1.0";
          format = "pyproject";
          src = ./bikipy-inspect;

          nativeBuildInputs = [ pkgs.python312.pkgs.hatchling ];

          propagatedBuildInputs = with pkgs.python312.pkgs; [
            numpy
            polars
            matplotlib
            seaborn
            moviepy
            opencv4
            click
          ];

          # nixpkgs' opencv4 provides cv2 under a different distribution name.
          pythonRemoveDeps = [ "opencv-python" ];
        };

        # Combined wrapper that puts both bkpy (Rust) and bkpy-inspect (Python)
        # on the same PATH, so the Rust CLI can invoke Python seamlessly.
        bikipy = pkgs.symlinkJoin {
          name = "bikipy";
          paths = [ bikipy-cli bikipy-inspect ];
        };
      in
      {
        checks = {
          inherit bikipy-cli;

          bikipy-clippy = craneLib.cargoClippy (
            commonArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            }
          );

          bikipy-doc = craneLib.cargoDoc (
            commonArgs
            // {
              inherit cargoArtifacts;
              env.RUSTDOCFLAGS = "--deny warnings";
            }
          );

          bikipy-fmt = craneLib.cargoFmt {
            inherit src;
          };

          bikipy-toml-fmt = craneLib.taploFmt {
            src = pkgs.lib.sources.sourceFilesBySuffices src [ ".toml" ];
          };

          bikipy-audit = craneLib.cargoAudit {
            inherit src advisory-db;
          };

          bikipy-deny = craneLib.cargoDeny {
            inherit src;
          };

          bikipy-nextest = craneLib.cargoNextest (
            commonArgs
            // {
              inherit cargoArtifacts;
              partitions = 1;
              partitionType = "count";
              cargoNextestPartitionsExtraArgs = "--no-tests=pass";
            }
          );

        };

        packages = {
          inherit bikipy-cli bikipy-inspect bikipy;
          default = bikipy;
        };

        apps = {
          bikipy = flake-utils.lib.mkApp {
            drv = bikipy;
          };
          default = self.apps.${system}.bikipy;
          push-flake-inputs = rs-harbor.lib.mkAtticPush {
            inherit pkgs;
            adapter = rs-harbor.lib.mkAdapter {
              attic = {
                endpoint = "https://attic.candee.baby";
                cache = "canix";
              };
            };
            flake = ".";
          };
        };

        devShells.default = craneLib.devShell {
          checks = self.checks.${system};

          packages = [
            bikipy-inspect
          ];
        };
      }
    );
}
