use vgonio::error::Error;

fn run() -> Result<(), Error> {
    use clap::Parser;
    use vgonio::app;
    use vgonio::app::VgonioArgs;

    let launch_time = std::time::SystemTime::now();

    // Parse the command line arguments
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
