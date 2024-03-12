use crate::app::{args::SubCommand, Config};
use base::error::VgonioError;

/// ANSI color codes.
pub mod ansi {
    pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
    pub const BRIGHT_RED: &str = "\u{001b}[31m";
    pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
    pub const RESET: &str = "\u{001b}[0m";

    pub const RED_EXCLAMATION: &str = "\u{001b}[31m!\u{001b}[0m";
    pub const CYAN_CHECK: &str = "\u{001b}[36m✓\u{001b}[0m";

    pub const YELLOW_GT: &str = "\u{001b}[33m>\u{001b}[0m";

    pub const CYAN_MINUS: &str = "\u{001b}[36m-\u{001b}[0m";
}

mod cmd_convert;

mod cmd_fit;
#[cfg(feature = "surf-gen")]
mod cmd_generate;
mod cmd_info;
mod cmd_measure;

pub use cmd_convert::ConvertOptions;
pub use cmd_fit::FitOptions;
#[cfg(feature = "surf-gen")]
pub use cmd_generate::GenerateOptions;
pub use cmd_measure::MeasureOptions;

/// Entry point of vgonio CLI.
pub fn run(cmd: SubCommand, config: Config) -> Result<(), VgonioError> {
    match cmd {
        SubCommand::Measure(opts) => cmd_measure::measure(opts, config),
        SubCommand::PrintInfo(opts) => cmd_info::print_info(opts, config),

        #[cfg(feature = "surf-gen")]
        SubCommand::Generate(opts) => cmd_generate::generate(opts, config),

        SubCommand::Convert(opts) => cmd_convert::convert(opts, config),

        SubCommand::Fit(opts) => cmd_fit::fit(opts, config),
    }
}
