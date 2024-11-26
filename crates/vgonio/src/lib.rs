#[derive(clap::Parser, Debug)]
#[clap(
    author,
    version,
    about = "Micro-geometry level light transport simulation tool.",
    arg_required_else_help(true)
)]
pub struct Args {
    #[clap(subcommand)]
    pub subcmd: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    Builtin(Builtin),
}

#[derive(clap::Args, Debug)]
pub struct Builtin {
    pub a: i32,
    pub b: i32,
}
