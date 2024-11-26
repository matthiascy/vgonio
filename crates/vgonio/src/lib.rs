#[derive(clap::Parser, Debug)]
#[clap(
    author,
    version,
    about = "Micro-geometry level light transport simulation tool."
)]
pub struct Args {
    #[clap(subcommand)]
    pub command: Subcommand,
}

#[derive(clap::Subcommand, Debug)]
pub enum Subcommand {
    #[cfg(feature = "compute")]
    #[clap(about = "Serve the VGonio computation service.")]
    Serve(Connection),
    #[cfg(feature = "compute")]
    #[clap(about = "Connect to the VGonio computation service.")]
    Connect(Connection),
}

#[derive(clap::Args, Debug)]
pub struct Connection {
    pub port: u16,
    pub addr: std::net::SocketAddr,
}
