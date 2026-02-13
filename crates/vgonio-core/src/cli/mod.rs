//! Common CLI utilities.

pub mod ansi;

use std::{
    borrow::Cow,
    io::{IsTerminal, Write},
    sync::{Arc, Mutex, OnceLock, RwLock},
};

/// Color emission policy for status output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Use color only when output stream is a terminal and `NO_COLOR` is not set.
    Auto,
    /// Always emit ANSI colors.
    Always,
    /// Never emit ANSI colors.
    Never,
}

/// Runtime config for CLI status output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrinterConfig {
    /// Suppresses status output on stdout.
    pub quiet: bool,
    /// Status verbosity level.
    ///
    /// `0` prints default status lines.
    /// Higher values can be used by `*_v` helpers for detailed output.
    pub verbosity: u8,
    /// Color output policy.
    pub color_mode: ColorMode,
}

impl Default for PrinterConfig {
    fn default() -> Self {
        Self {
            quiet: false,
            verbosity: 0,
            color_mode: ColorMode::Auto,
        }
    }
}

/// Configures runtime status output behavior.
pub fn setup_printer(config: PrinterConfig) {
    let lock = PRINTER_CONFIG.get_or_init(|| RwLock::new(PrinterConfig::default()));
    *lock.write().unwrap() = config;
}

/// Returns current runtime status output config.
pub fn printer_config() -> PrinterConfig {
    *PRINTER_CONFIG
        .get_or_init(|| RwLock::new(PrinterConfig::default()))
        .read()
        .unwrap()
}

static PRINTER_CONFIG: OnceLock<RwLock<PrinterConfig>> = OnceLock::new();

/// Output stream for status messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusStream {
    /// Standard output stream.
    Stdout,
    /// Standard error stream.
    Stderr,
}

/// A single status emission request passed to a [`StatusSink`].
pub struct StatusRequest<'a> {
    /// Which stream to target.
    pub stream: StatusStream,
    /// Prefix marker symbol.
    pub symbol: ansi::ColoredSymbol,
    /// Left indentation in spaces.
    pub indent: u32,
    /// Message payload.
    pub msg: std::fmt::Arguments<'a>,
    /// Minimum verbosity required for stdout messages.
    pub required_verbosity: u8,
    /// Whether to append a trailing newline.
    pub newline: bool,
    /// Whether to flush the target stream after writing.
    pub flush: bool,
}

/// Pluggable sink for user-facing status output.
pub trait StatusSink: Send + Sync {
    /// Emits one status message.
    fn emit(&self, request: StatusRequest<'_>);
}

/// Default status sink writing to terminal streams.
#[derive(Debug, Clone, Copy)]
pub struct CliSink {
    serialize_writes: bool,
}

impl Default for CliSink {
    fn default() -> Self { Self::new() }
}

impl CliSink {
    /// Creates a CLI sink with line-buffered direct writes.
    pub const fn new() -> Self {
        Self {
            serialize_writes: false,
        }
    }

    /// Enables or disables global write serialization.
    pub const fn with_serialized_writes(mut self, enabled: bool) -> Self {
        self.serialize_writes = enabled;
        self
    }

    fn emit_unlocked(&self, request: StatusRequest<'_>) {
        if matches!(request.stream, StatusStream::Stdout)
            && !stdout_enabled(request.required_verbosity)
        {
            return;
        }

        let indent = indent_str(request.indent);
        let colors = color_enabled(request.stream);

        match request.stream {
            StatusStream::Stdout => {
                let mut out = std::io::stdout().lock();
                if colors {
                    if request.newline {
                        let _ = writeln!(out, "{}{} {}", indent, request.symbol, request.msg);
                    } else {
                        let _ = write!(out, "{}{} {}", indent, request.symbol, request.msg);
                    }
                } else if request.newline {
                    let _ = writeln!(out, "{}{} {}", indent, request.symbol.symbol, request.msg);
                } else {
                    let _ = write!(out, "{}{} {}", indent, request.symbol.symbol, request.msg);
                }
                if request.flush {
                    let _ = out.flush();
                }
            },
            StatusStream::Stderr => {
                let mut out = std::io::stderr().lock();
                if colors {
                    if request.newline {
                        let _ = writeln!(out, "{}{} {}", indent, request.symbol, request.msg);
                    } else {
                        let _ = write!(out, "{}{} {}", indent, request.symbol, request.msg);
                    }
                } else if request.newline {
                    let _ = writeln!(out, "{}{} {}", indent, request.symbol.symbol, request.msg);
                } else {
                    let _ = write!(out, "{}{} {}", indent, request.symbol.symbol, request.msg);
                }
                if request.flush {
                    let _ = out.flush();
                }
            },
        }
    }
}

impl StatusSink for CliSink {
    fn emit(&self, request: StatusRequest<'_>) {
        if self.serialize_writes {
            let _guard = SINK_WRITE_LOCK
                .get_or_init(|| Mutex::new(()))
                .lock()
                .unwrap();
            self.emit_unlocked(request);
        } else {
            self.emit_unlocked(request);
        }
    }
}

/// No-op status sink.
#[derive(Debug, Default, Clone, Copy)]
pub struct SilentSink;

impl StatusSink for SilentSink {
    fn emit(&self, _request: StatusRequest<'_>) {}
}

static STATUS_SINK: OnceLock<RwLock<Arc<dyn StatusSink>>> = OnceLock::new();
static SINK_WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn default_status_sink() -> Arc<dyn StatusSink> { Arc::new(CliSink::new()) as Arc<dyn StatusSink> }

fn status_sink() -> Arc<dyn StatusSink> {
    STATUS_SINK
        .get_or_init(|| RwLock::new(default_status_sink()))
        .read()
        .unwrap()
        .clone()
}

/// Installs a custom status sink.
pub fn set_status_sink<S>(sink: S)
where
    S: StatusSink + 'static,
{
    let lock = STATUS_SINK.get_or_init(|| RwLock::new(default_status_sink()));
    *lock.write().unwrap() = Arc::new(sink);
}

/// Restores the default CLI sink.
pub fn reset_status_sink() {
    let lock = STATUS_SINK.get_or_init(|| RwLock::new(default_status_sink()));
    *lock.write().unwrap() = default_status_sink();
}

fn dispatch_status(request: StatusRequest<'_>) { status_sink().emit(request); }

fn indent_str(indent: u32) -> Cow<'static, str> {
    match indent {
        0 => Cow::Borrowed(""),
        2 => Cow::Borrowed("  "),
        4 => Cow::Borrowed("    "),
        6 => Cow::Borrowed("      "),
        8 => Cow::Borrowed("        "),
        _ => Cow::Owned(" ".repeat(indent as usize)),
    }
}

fn color_enabled(stream: StatusStream) -> bool {
    let config = printer_config();
    match config.color_mode {
        ColorMode::Always => true,
        ColorMode::Never => false,
        ColorMode::Auto => {
            if std::env::var_os("NO_COLOR").is_some() {
                return false;
            }
            match stream {
                StatusStream::Stdout => std::io::stdout().is_terminal(),
                StatusStream::Stderr => std::io::stderr().is_terminal(),
            }
        },
    }
}

fn stdout_enabled(required_verbosity: u8) -> bool {
    let config = printer_config();
    !config.quiet && config.verbosity >= required_verbosity
}

/// Parses the arguments, returns the arguments and the launch time.
///
/// # Arguments
///
/// * `name` - The name of the program.
///
/// # Returns
///
/// * `args` - The parsed arguments.
/// * `launch_time` - The launch time of the program.
pub fn parse_args<T: clap::Parser>(name: &str) -> (T, std::time::SystemTime) {
    let args = T::parse();
    let launch_time = std::time::SystemTime::now();
    log::info!(
        "{} launched at {} on {}.",
        name,
        chrono::DateTime::<chrono::Utc>::from(launch_time),
        std::env::consts::OS
    );

    (args, launch_time)
}

/// Common arguments for the CLI.
#[derive(Debug, Copy, Clone, clap::Args)]
pub struct CommonArgs {
    #[clap(
        long,
        help = "The log level. 0 = Error, 1 = Warn, 2 = Info, 3 = Debug, 4 = Trace",
        default_value = "0"
    )]
    /// The log level.
    pub log_level: u8,
}

/// A filter for the logger.
///
/// This is a tuple of a module name and a log level filter.
pub type LogFilter<'a> = (&'a str, log::LevelFilter);

/// Initialises logging settings.
///
/// # Arguments
///
/// * `timestamp` - Whether to print the timestamp in the log; This is the base time for the
///   timestamp.
/// * `log_level` - The log level to filter. This is the top level log level for the program. See
///   [`log_filter_from_level`] for more details.
/// * `filters` - The filters to apply to the logger.
pub fn setup_logging(
    timestamp: Option<std::time::SystemTime>,
    log_level: u8,
    filters: &[LogFilter],
) {
    use std::io::Write;
    let mut builder = env_logger::builder();
    builder.format(move |buf, record| {
        let top_level_module = record.module_path().unwrap().split("::").next().unwrap();
        match timestamp {
            Some(timestamp) => {
                let duration = timestamp.elapsed().unwrap();
                let millis = duration.as_millis() % 1000;
                let seconds = duration.as_secs() % 60;
                let minutes = (duration.as_secs() / 60) % 60;
                let hours = (duration.as_secs() / 60) / 60;
                writeln!(
                    buf,
                    "{}:{}:{}.{:03} {:5} [{}]: {}",
                    hours,
                    minutes,
                    seconds,
                    millis,
                    record.level(),
                    top_level_module,
                    record.args()
                )
            },
            None => {
                writeln!(
                    buf,
                    "{:5} [{}]: {}",
                    record.level(),
                    top_level_module,
                    record.args()
                )
            },
        }
    });
    for (module, level) in filters {
        builder.filter(Some(module), *level);
    }
    builder
        .filter_level(log_filter_from_level(log_level))
        .init();
}

/// Converts a log level to a log filter.
pub fn log_filter_from_level(level: u8) -> log::LevelFilter {
    match level {
        0 => log::LevelFilter::Error,
        1 => log::LevelFilter::Warn,
        2 => log::LevelFilter::Info,
        3 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    }
}

/// Prints a message with a colored symbol to stdout.
pub fn print(symbol: ansi::ColoredSymbol, indent: u32, msg: std::fmt::Arguments) {
    print_v(symbol, indent, msg, 0);
}

/// Prints a message with a colored symbol to stdout without a trailing newline.
pub fn print_inline(symbol: ansi::ColoredSymbol, indent: u32, msg: std::fmt::Arguments) {
    print_inline_v(symbol, indent, msg, 0);
}

/// Prints a message with a colored symbol to stderr.
pub fn eprint(symbol: ansi::ColoredSymbol, indent: u32, msg: std::fmt::Arguments) {
    dispatch_status(StatusRequest {
        stream: StatusStream::Stderr,
        symbol,
        indent,
        msg,
        required_verbosity: 0,
        newline: true,
        flush: false,
    });
}

/// Prints an informational step marker (`>`).
pub fn step(indent: u32, msg: std::fmt::Arguments) { print(ansi::YELLOW_GT, indent, msg); }

/// Prints an informational step marker (`>`) if verbosity is high enough.
pub fn step_v(required_verbosity: u8, indent: u32, msg: std::fmt::Arguments) {
    print_v(ansi::YELLOW_GT, indent, msg, required_verbosity);
}

/// Prints an informational step marker (`>`) without a trailing newline.
pub fn step_inline(indent: u32, msg: std::fmt::Arguments) {
    print_inline(ansi::YELLOW_GT, indent, msg);
}

/// Prints an informational step marker (`>`) without a trailing newline if verbosity is high
/// enough.
pub fn step_inline_v(required_verbosity: u8, indent: u32, msg: std::fmt::Arguments) {
    print_inline_v(ansi::YELLOW_GT, indent, msg, required_verbosity);
}

/// Prints an informational step marker (`>`) and flushes stdout.
pub fn step_inline_flush(indent: u32, msg: std::fmt::Arguments) {
    print_inline_v_internal(ansi::YELLOW_GT, indent, msg, 0, true);
}

/// Prints an informational step marker (`>`) with flush if verbosity is high enough.
pub fn step_inline_flush_v(required_verbosity: u8, indent: u32, msg: std::fmt::Arguments) {
    print_inline_v_internal(ansi::YELLOW_GT, indent, msg, required_verbosity, true);
}

/// Prints an informational note marker (`-`).
pub fn note(indent: u32, msg: std::fmt::Arguments) { print(ansi::CYAN_MINUS, indent, msg); }

/// Prints an informational note marker (`-`) if verbosity is high enough.
pub fn note_v(required_verbosity: u8, indent: u32, msg: std::fmt::Arguments) {
    print_v(ansi::CYAN_MINUS, indent, msg, required_verbosity);
}

/// Prints a success marker (`✓`).
pub fn success(indent: u32, msg: std::fmt::Arguments) { print(ansi::CYAN_CHECK, indent, msg); }

/// Prints a success marker (`✓`) if verbosity is high enough.
pub fn success_v(required_verbosity: u8, indent: u32, msg: std::fmt::Arguments) {
    print_v(ansi::CYAN_CHECK, indent, msg, required_verbosity);
}

/// Prints an error marker (`!`) to stderr.
pub fn error(indent: u32, msg: std::fmt::Arguments) { eprint(ansi::RED_EXCLAMATION, indent, msg); }

fn print_inline_v_internal(
    symbol: ansi::ColoredSymbol,
    indent: u32,
    msg: std::fmt::Arguments,
    required_verbosity: u8,
    flush: bool,
) {
    if !stdout_enabled(required_verbosity) {
        return;
    }
    dispatch_status(StatusRequest {
        stream: StatusStream::Stdout,
        symbol,
        indent,
        msg,
        required_verbosity,
        newline: false,
        flush,
    });
}

/// Prints a message with a colored symbol to stdout if verbosity is high enough.
pub fn print_v(
    symbol: ansi::ColoredSymbol,
    indent: u32,
    msg: std::fmt::Arguments,
    required_verbosity: u8,
) {
    if !stdout_enabled(required_verbosity) {
        return;
    }
    dispatch_status(StatusRequest {
        stream: StatusStream::Stdout,
        symbol,
        indent,
        msg,
        required_verbosity,
        newline: true,
        flush: false,
    });
}

/// Prints a message with a colored symbol to stdout without a trailing newline if verbosity is
/// high enough.
pub fn print_inline_v(
    symbol: ansi::ColoredSymbol,
    indent: u32,
    msg: std::fmt::Arguments,
    required_verbosity: u8,
) {
    print_inline_v_internal(symbol, indent, msg, required_verbosity, false);
}

/// Prints a message to the console.
#[deprecated(note = "Use cli::step/note/success/error or cli::print/print_v instead.")]
pub fn println(symbol: char, indent: u32, msg: std::fmt::Arguments, color: ansi::Color) {
    print(ansi::ColoredSymbol::new(color, symbol), indent, msg);
}
