pub mod app;
pub mod error;
pub mod height_field;
pub mod isect;
mod math;
pub mod mesh;

use crate::error::Error;

#[cfg(not(target_arch = "wasm32"))]
pub fn run() -> Result<(), Error> {
    use app::VgonioArgs;
    use clap::Parser;

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
