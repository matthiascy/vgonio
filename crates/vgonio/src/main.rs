#![warn(clippy::all, rust_2021_compatibility)]

use clap::Parser;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    match vgonio::Args::try_parse() {
        Ok(args) => match args.subcmd {
            vgonio::Command::Builtin(vgonio::Builtin { a, b }) => {
                println!("VGonio v{}", env!("CARGO_PKG_VERSION"));
                println!("  by {}", env!("CARGO_PKG_AUTHORS"));
                println!();
                println!("  Running with the following arguments:");
                println!("    {:?}", args);
            },
        },
        Err(_) => {
            let args: Box<[String]> = std::env::args().collect();

            if args.len() < 2 {
                vgonio::Args::parse_from(&["vgonio", "--help"]);
                std::process::exit(1);
            }

            let subcmd = &args[1];
            let subcmd_binary = format!("vgonio-{}", subcmd);
            let exe_path = std::env::current_exe()
                .expect("Failed to get current executable path");
            let exe_dir = exe_path
                .parent()
                .expect("Failed to get parent directory of executable");
            let new_path = format!(
                "{}:{}",
                exe_path.to_str().unwrap(),
                std::env::var("PATH").unwrap_or_default()
            );
            std::env::set_var("PATH", new_path);

            if let Ok(full_path) = which::which(&subcmd_binary) {
                let mut cmd = std::process::Command::new(full_path);
                cmd.args(&args[2..]);

                let status = cmd.status().expect("Failed to execute subcommand");

                std::process::exit(status.code().unwrap_or(1));
            } else {
                eprintln!("Unknown subcommand: {}", subcmd);
                std::process::exit(1);
            }
        },
    }

    Ok(())
}
