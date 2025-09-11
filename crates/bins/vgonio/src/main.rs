#![warn(clippy::all, rust_2021_compatibility)]

use clap::{CommandFactory, Parser};
use vgonio::Args;

fn alias(cmd: &str) -> &str {
    match cmd {
        "serve" | "daemon" => "srv",
        "measure" | "acquire" | "simulate" => "sim",
        "view" => "viz",
        "plot" | "chart" | "graph" => "plt",
        other => other,
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::try_parse();

    match args {
        Ok(args) => match args.subcmd {
            vgonio::Command::List => {
                let subcmds = vgonio::list_all_external_commands(&vgonio::search_paths());

                for (cmd, path) in subcmds {
                    println!("{cmd}: {}", path.display());
                }
            },
        },
        Err(err) => {
            let args: Box<[String]> = std::env::args().collect();

            if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
                let mut cmd = Args::command();
                cmd.print_help()?;
                println!();
                std::process::exit(0);
            }

            // Try to find and execute the external subcommand.
            let wanted = alias(&args[1]);
            let subcmds = vgonio::list_all_external_commands(&vgonio::search_paths());

            if let Some(path) = subcmds.get(wanted) {
                let status = std::process::Command::new(path)
                    .args(&args[2..])
                    .status()?;
                std::process::exit(status.code().unwrap_or(1));
            } else {
                err.print()?;
                eprintln!("\nTip: run 'vgn list' to see all available subcommands.");
                std::process::exit(1);
            }
        },
    }

    Ok(())
}
