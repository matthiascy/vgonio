use vgonio_core::{cli, cli::CommonArgs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (args, launch_time) = cli::parse_args::<CliArgs>("vgonio-measure");

    cli::setup_logging(Some(launch_time), args.common.log_level, &[]);

    Ok(())
}

#[derive(clap::Parser, Debug, Clone)]
#[clap(
    author,
    version,
    about = "Micro-geometry level light transportation simulation (measure)."
)]
pub struct CliArgs {
    #[command(flatten)]
    pub common: CommonArgs,
}
