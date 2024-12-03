#![warn(clippy::all, rust_2021_compatibility)]

use clap::Parser;
use vgonio::Args;

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::try_parse();

    match args {
        Ok(args) => match args.subcmd {
            vgonio::Command::List => {
                let search_paths = vgonio::search_paths();
                let subcmds = vgonio::list_all_external_commands(&search_paths);

                for (name, path) in subcmds {
                    println!("{}: {}", name, path.display());
                }
            },
        },
        Err(err) => {
            let args: Box<[String]> = std::env::args().collect();

            if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
                Args::parse_from(&["vgonio", "--help"]);
                std::process::exit(1);
            }

            let subcmd = format!("vgonio-{}", &args[1]);
            let search_paths = vgonio::search_paths();
            let subcmds = vgonio::list_all_external_commands(&search_paths);

            if subcmds.contains_key(&subcmd) {
                let mut cmd = std::process::Command::new(&subcmds[&subcmd]);
                cmd.args(&args[2..]);
                let status = cmd.status().expect("Failed to execute subcommand");
                std::process::exit(status.code().unwrap_or(1));
            } else {
                let _ = err.print();
                std::process::exit(1);
            }
        },
    }

    Ok(())
}
