use crate::error::Error;

mod app;
mod error;
mod state;

fn run() -> Result<(), Error> {
    // Parse the command line arguments
    use crate::app::VgonioArgs;
    use clap::Parser;

    let launch_time = std::time::SystemTime::now();

    let args: VgonioArgs = VgonioArgs::parse();

    // Initialize vgonio application
    app::init(&args, launch_time);

    // Dispatch subcommands
    match args.command {
        None => app::launch_gui_client(),
        Some(cmd) => app::execute_command(cmd),
    }
}

fn main() {
    ::std::process::exit(match run() {
        Ok(_) => 0,
        Err(ref e) => {
            eprintln!("{}", e);
            1
        }
    })
}
