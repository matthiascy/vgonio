//! CLI output and status reporting system.
//!
//! This module provides a unified interface for user-facing status output (progress indicators,
//! completion messages, errors) with support for verbosity gating, color policies, and pluggable
//! output sinks.
//!
//! ## Architecture
//!
//! The system separates two output channels:
//!
//! - **Diagnostic logs**: Use the `log` crate (`log::info!`, `log::debug!`, etc.) for
//!   developer-oriented diagnostic output.
//! - **Status output**: Use this module's functions (`step`, `note`, `success`, `error`) for
//!   user-facing progress and status messages.
//!
//! This separation allows independent control over technical diagnostics vs. UX-oriented feedback.
//!
//! ## Configuration
//!
//! Status output behavior is controlled by [`PrinterConfig`], which must be set up early during
//! application startup via [`setup_printer`]:
//!
//! ```no_run
//! use vgn_core::cli::{setup_printer, ColorMode, PrinterConfig};
//!
//! setup_printer(PrinterConfig {
//!     quiet: false,                // Allow stdout status output
//!     verbosity: 1,                // Show level 0 and 1 messages
//!     color_mode: ColorMode::Auto, // Auto-detect terminal color support
//! });
//! ```
//!
//! ### Verbosity Levels
//!
//! - **Level 0** (default): High-level progress and outcomes
//! - **Level 1+**: Detailed operational status, typically enabled via `--verbose`
//!
//! Functions ending in `_v` accept a `required_verbosity` parameter to gate output:
//!
//! ```no_run
//! use vgn_core::cli;
//!
//! cli::step(0, format_args!("Starting measurement...")); // Always shown
//! cli::step_v(1, 2, format_args!("Allocating buffers...")); // Only if verbosity >= 1
//! ```
//!
//! ### Color Policy
//!
//! [`ColorMode`] controls ANSI color emission:
//! - `Auto`: Colors enabled when output is a terminal and `NO_COLOR` is unset
//! - `Always`: Force colors regardless of environment
//! - `Never`: Disable all colors
//!
//! ## Status Output API
//!
//! ### Primary Functions
//!
//! - [`step`] / [`step_v`]: Informational step marker (`>`)
//! - [`note`] / [`note_v`]: Supplementary note marker (`-`)
//! - [`success`] / [`success_v`]: Success marker (`✓`)
//! - [`warning`] / [`warning_v`]: Warning marker (`⚠`)
//! - [`error`]: Error marker (`!`) - always emitted to stderr
//!
//! ### Utility Functions
//!
//! - [`format_duration`]: Format `Duration` as human-readable string (e.g., "1h 2m 3s")
//! - [`format_duration_secs`]: Format `Duration` as fractional seconds (e.g., "1.50s")
//! - [`format_bytes`]: Format byte sizes (e.g., "2.50 MiB")
//!
//! ### Ergonomic Macros
//!
//! For cleaner syntax and zero-cost verbosity gating, use the macro variants:
//!
//! ```no_run
//! # use vgn_core::{cli::Indent, cli_note, cli_step, cli_timed, cli_warning};
//! # let count = 5;
//!
//! cli_step!(Indent::ROOT, "Starting process...");
//! cli_note!(Indent::SECTION, "Found {} items", count);
//! cli_warning!(Indent::DETAIL, "Deprecated feature in use");
//!
//! // Time a block of code
//! cli_timed!(Indent::ROOT, "Processing data", {
//!     // expensive operation
//! });
//! ```
//!
//! The `_v` macro variants only evaluate arguments when verbosity permits:
//!
//! ```no_run
//! # use vgn_core::cli_step_v;
//! # let expensive_debug = || "expensive debug info".to_string();
//!
//! // expensive_debug() only runs if verbosity >= 1
//! cli_step_v!(1, 2u32, "Debug info: {}", expensive_debug());
//! ```
//!
//! ### Type-Safe Indentation
//!
//! Use the [`Indent`] type for semantic, self-documenting indentation:
//!
//! ```no_run
//! use vgn_core::cli::Indent;
//!
//! let section = Indent::SECTION; // 2 spaces
//! let nested = section.nest(); // 4 spaces
//! let custom = Indent::custom(10); // 10 spaces
//! ```
//!
//! Legacy `u32` constants are also available in the [`indent`] module.
//!
//! ### Inline Output
//!
//! For same-line updates (e.g., progress indicators without newlines):
//!
//! - [`step_inline`] / [`step_inline_v`]: Without automatic flush
//! - [`step_inline_flush`] / [`step_inline_flush_v`]: With immediate flush
//!
//! ## Output Sinks
//!
//! The system uses pluggable [`StatusSink`] implementations to route status output:
//!
//! - [`CliSink`] (default): Writes to stdout/stderr with optional serialization
//! - [`SilentSink`]: Discards all output (useful for `--quiet` mode)
//!
//! ### Sink Selection
//!
//! ```no_run
//! use vgn_core::cli::{set_status_sink, use_cli_sink, SilentSink};
//!
//! // Use CLI sink with serialized writes (prevents interleaving)
//! use_cli_sink(true);
//!
//! // Or install a custom sink
//! set_status_sink(SilentSink);
//! ```
//!
//! Serialized writes add overhead but guarantee deterministic output order when multiple threads
//! emit status messages concurrently.
//!
//! ## Example Usage
//!
//! ### Using Functions
//!
//! ```no_run
//! # use vgn_core::cli::{self, setup_printer, ColorMode, PrinterConfig};
//!
//! // Setup during app initialization
//! setup_printer(PrinterConfig {
//!     quiet: false,
//!     verbosity: 0,
//!     color_mode: ColorMode::Auto,
//! });
//!
//! // User-facing status output
//! cli::step(0, format_args!("Loading configuration..."));
//! cli::success(0, format_args!("Configuration loaded"));
//!
//! // Detailed status (only shown if verbosity >= 1)
//! cli::note_v(1, 2, format_args!("Using default cache directory"));
//!
//! // Warning for non-fatal issues
//! cli::warning(0, format_args!("Deprecated option detected"));
//!
//! // Error output (always to stderr)
//! cli::error(0, format_args!("Failed to open file: permission denied"));
//! ```
//!
//! ### Using Macros (Recommended)
//!
//! ```no_run
//! # use vgn_core::{
//! #    cli::{self, setup_printer, ColorMode, Indent, PrinterConfig},
//! #    cli_error, cli_note_v, cli_step, cli_success, cli_warning,
//! # };
//!
//! setup_printer(PrinterConfig {
//!     quiet: false,
//!     verbosity: 1,
//!     color_mode: ColorMode::Auto,
//! });
//!
//! // Cleaner syntax with macros
//! cli_step!(Indent::ROOT, "Loading configuration...");
//! cli_success!(Indent::ROOT, "Configuration loaded");
//!
//! # let get_cache_dir = || "/path/to/cache";
//!
//! // Zero-cost verbosity gating - only evaluates if verbosity >= 1
//! cli_note_v!(1, Indent::SECTION, "Cache dir: {}", get_cache_dir());
//!
//! # let path = "/path/to/file";
//!
//! cli_warning!(Indent::ROOT, "Using deprecated feature");
//! cli_error!(Indent::ROOT, "Failed to open file: {}", path);
//! ```

pub mod ansi;

use std::{
    borrow::Cow,
    io::{IsTerminal, Write},
    sync::{Arc, Mutex, OnceLock, RwLock},
};

/// Type-safe indentation for status output.
///
/// Provides semantic constants for common indentation levels and prevents
/// accidental use of arbitrary values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Indent(u32);

impl Indent {
    /// Root level (no indentation).
    pub const ROOT: Self = Self(0);
    /// Section level (2 spaces).
    pub const SECTION: Self = Self(2);
    /// Subsection level (4 spaces).
    pub const SUBSECTION: Self = Self(4);
    /// Detail level (6 spaces).
    pub const DETAIL: Self = Self(6);
    /// Deep detail level (8 spaces).
    pub const DEEP: Self = Self(8);

    /// Creates a custom indentation level.
    pub const fn custom(spaces: u32) -> Self { Self(spaces) }

    /// Creates an indentation one level deeper (adds 2 spaces).
    pub const fn nest(self) -> Self { Self(self.0 + 2) }

    /// Returns the indentation as a number of spaces.
    pub const fn as_u32(self) -> u32 { self.0 }
}

impl From<u32> for Indent {
    fn from(spaces: u32) -> Self { Self(spaces) }
}

impl From<Indent> for u32 {
    fn from(indent: Indent) -> Self { indent.0 }
}

impl Default for Indent {
    fn default() -> Self { Self::ROOT }
}

impl std::fmt::Display for Indent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}sp", self.0) }
}

/// Formats a duration in a human-readable way.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use vgn_core::cli::format_duration;
///
/// assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
/// assert_eq!(format_duration(Duration::from_secs(75)), "1m 15s");
/// assert_eq!(format_duration(Duration::from_secs(3661)), "1h 1m 1s");
/// ```
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_secs = duration.as_secs();
    let millis = duration.subsec_millis();

    if total_secs == 0 {
        if millis == 0 {
            let micros = duration.subsec_micros();
            if micros == 0 {
                return format!("{}ns", duration.subsec_nanos());
            }
            return format!("{}μs", micros);
        }
        return format!("{}ms", millis);
    }

    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    let mut parts = Vec::new();
    if hours > 0 {
        parts.push(format!("{}h", hours));
    }
    if minutes > 0 {
        parts.push(format!("{}m", minutes));
    }
    if seconds > 0 || parts.is_empty() {
        if millis > 0 && parts.is_empty() {
            parts.push(format!("{}.{:03}s", seconds, millis));
        } else {
            parts.push(format!("{}s", seconds));
        }
    }

    parts.join(" ")
}

/// Formats a duration as fractional seconds.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use vgn_core::cli::format_duration_secs;
///
/// assert_eq!(
///     format_duration_secs(Duration::from_millis(1500), 2),
///     "1.50s"
/// );
/// assert_eq!(format_duration_secs(Duration::from_secs(42), 0), "42s");
/// ```
pub fn format_duration_secs(duration: std::time::Duration, precision: usize) -> String {
    format!("{:.prec$}s", duration.as_secs_f64(), prec = precision)
}

/// Formats a byte size in a human-readable way.
///
/// # Examples
///
/// ```
/// use vgn_core::cli::format_bytes;
///
/// assert_eq!(format_bytes(512), "512 B");
/// assert_eq!(format_bytes(2048), "2.00 KiB");
/// assert_eq!(format_bytes(1_048_576), "1.00 MiB");
/// ```
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB", "PiB"];

    if bytes < 1024 {
        return format!("{} B", bytes);
    }

    let mut size = bytes as f64;
    let mut unit_idx = 0;

    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }

    format!("{:.2} {}", size, UNITS[unit_idx])
}

/// Legacy indentation constants (consider using [`Indent`] type instead).
pub mod indent {
    /// Root level indentation (0 spaces).
    pub const ROOT: u32 = 0;
    /// Section level indentation (2 spaces).
    pub const SECTION: u32 = 2;
    /// Subsection level indentation (4 spaces).
    pub const SUBSECTION: u32 = 4;
    /// Detail level indentation (6 spaces).
    pub const DETAIL: u32 = 6;
    /// Deep detail level indentation (8 spaces).
    pub const DEEP: u32 = 8;
}

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
///
/// This struct carries all information needed to render one status message,
/// including stream selection, formatting, verbosity gating, and flush behavior.
pub struct StatusRequest<'a> {
    /// Which stream to target (stdout or stderr).
    pub stream: StatusStream,
    /// Prefix marker symbol with color.
    pub symbol: ansi::ColoredSymbol,
    /// Left indentation in spaces.
    pub indent: u32,
    /// Message payload (use `format_args!` to construct).
    pub msg: std::fmt::Arguments<'a>,
    /// Minimum verbosity level required for this message to be emitted to stdout.
    /// Stderr messages ignore this field.
    pub required_verbosity: u8,
    /// Whether to append a trailing newline after the message.
    pub newline: bool,
    /// Whether to flush the target stream immediately after writing.
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
///
/// **Warning:** Replacing the sink during active status output from multiple threads
/// may cause interleaved output or lost messages in flight. Best practice is to
/// install sinks during application startup before spawning parallel tasks.
///
/// # Example
///
/// ```no_run
/// use vgn_core::cli::{set_status_sink, SilentSink};
/// set_status_sink(SilentSink);
/// ```
pub fn set_status_sink<S>(sink: S)
where
    S: StatusSink + 'static,
{
    let lock = STATUS_SINK.get_or_init(|| RwLock::new(default_status_sink()));
    *lock.write().unwrap() = Arc::new(sink);
}

/// Replaces the current sink with the built-in CLI sink, optionally serializing writes.
///
/// When `serialized_writes` is `true`, all status output is serialized through a global
/// mutex, preventing interleaving from parallel tasks at the cost of performance.
/// Enable this when deterministic, sequential output is required (e.g., when piped or
/// for automated testing).
///
/// # Example
///
/// ```no_run
/// use vgn_core::cli::use_cli_sink;
/// // Enable serialized writes for piped output
/// use_cli_sink(true);
/// ```
pub fn use_cli_sink(serialized_writes: bool) {
    set_status_sink(CliSink::new().with_serialized_writes(serialized_writes));
}

/// Restores the default CLI sink (non-serialized).
///
/// This resets any custom sink back to the default [`CliSink`] with line-buffered writes.
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

/// Prints a warning marker (`⚠`) to stdout.
pub fn warning(indent: u32, msg: std::fmt::Arguments) { print(ansi::YELLOW_WARN, indent, msg); }

/// Prints a warning marker (`⚠`) to stdout if verbosity is high enough.
pub fn warning_v(required_verbosity: u8, indent: u32, msg: std::fmt::Arguments) {
    print_v(ansi::YELLOW_WARN, indent, msg, required_verbosity);
}

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

/// Ergonomic macros for CLI status output.
///
/// These macros provide a cleaner syntax and zero-cost verbosity gating.
/// Arguments are only evaluated when the output will actually be emitted.

/// Prints an informational step marker (`>`).
///
/// # Examples
///
/// ```ignore,no_run
/// # use vgn_core::{cli_step, cli::Indent};
/// # let count = 5;
/// # let name = "example.txt";
///
/// cli_step!(0u32, "Loading configuration...");
/// cli_step!(2u32, "Found {} files", count);
/// cli_step!(Indent::SECTION, "Processing {}", name);
/// ```
#[macro_export]
macro_rules! cli_step {
    ($indent:expr, $($arg:tt)*) => {
        $crate::cli::step($crate::cli::indent_as_u32($indent), format_args!($($arg)*))
    };
}

/// Prints an informational step marker (`>`) if verbosity is high enough.
///
/// Arguments are only evaluated if verbosity requirement is met (zero-cost gating).
#[macro_export]
macro_rules! cli_step_v {
    ($verbosity:expr, $indent:expr, $($arg:tt)*) => {
        if $crate::cli::printer_config().verbosity >= $verbosity {
            $crate::cli::step_v($verbosity, $crate::cli::indent_as_u32($indent), format_args!($($arg)*))
        }
    };
}

/// Prints an informational note marker (`-`).
#[macro_export]
macro_rules! cli_note {
    ($indent:expr, $($arg:tt)*) => {
        $crate::cli::note($crate::cli::indent_as_u32($indent), format_args!($($arg)*))
    };
}

/// Prints an informational note marker (`-`) if verbosity is high enough.
#[macro_export]
macro_rules! cli_note_v {
    ($verbosity:expr, $indent:expr, $($arg:tt)*) => {
        if $crate::cli::printer_config().verbosity >= $verbosity {
            $crate::cli::note_v($verbosity, $crate::cli::indent_as_u32($indent), format_args!($($arg)*))
        }
    };
}

/// Prints a success marker (`✓`).
#[macro_export]
macro_rules! cli_success {
    ($indent:expr, $($arg:tt)*) => {
        $crate::cli::success($crate::cli::indent_as_u32($indent), format_args!($($arg)*))
    };
}

/// Prints a success marker (`✓`) if verbosity is high enough.
#[macro_export]
macro_rules! cli_success_v {
    ($verbosity:expr, $indent:expr, $($arg:tt)*) => {
        if $crate::cli::printer_config().verbosity >= $verbosity {
            $crate::cli::success_v($verbosity, $crate::cli::indent_as_u32($indent), format_args!($($arg)*))
        }
    };
}

/// Prints a warning marker (`⚠`).
#[macro_export]
macro_rules! cli_warning {
    ($indent:expr, $($arg:tt)*) => {
        $crate::cli::warning($crate::cli::indent_as_u32($indent), format_args!($($arg)*))
    };
}

/// Prints a warning marker (`⚠`) if verbosity is high enough.
#[macro_export]
macro_rules! cli_warning_v {
    ($verbosity:expr, $indent:expr, $($arg:tt)*) => {
        if $crate::cli::printer_config().verbosity >= $verbosity {
            $crate::cli::warning_v($verbosity, $crate::cli::indent_as_u32($indent), format_args!($($arg)*))
        }
    };
}

/// Prints an error marker (`!`) to stderr.
#[macro_export]
macro_rules! cli_error {
    ($indent:expr, $($arg:tt)*) => {
        $crate::cli::error($crate::cli::indent_as_u32($indent), format_args!($($arg)*))
    };
}

/// Prints an informational step marker (`>`) without a trailing newline.
#[macro_export]
macro_rules! cli_step_inline {
    ($indent:expr, $($arg:tt)*) => {
        $crate::cli::step_inline($crate::cli::indent_as_u32($indent), format_args!($($arg)*))
    };
}

/// Prints an informational step marker (`>`) without newline, with flush.
#[macro_export]
macro_rules! cli_step_inline_flush {
    ($indent:expr, $($arg:tt)*) => {
        $crate::cli::step_inline_flush($crate::cli::indent_as_u32($indent), format_args!($($arg)*))
    };
}

/// Times a block of code and prints the elapsed duration.
///
/// # Examples
///
/// ```no_run,ignore
/// # use vgn_core::{cli::Indent, cli_timed};
///
/// cli_timed!(Indent::ROOT, "Processing data", {
///     // expensive operation
///     std::thread::sleep(std::time::Duration::from_millis(100));
/// });
/// // Prints: "> Processing data... (100ms)"
/// ```
#[macro_export]
macro_rules! cli_timed {
    ($indent:expr, $msg:expr, $block:block) => {{
        $crate::cli_step_inline_flush!($indent, "{}...", $msg);
        let __start = std::time::Instant::now();
        let __result = $block;
        let __elapsed = __start.elapsed();
        println!(" ({})", $crate::cli::format_duration(__elapsed));
        __result
    }};
}

/// Times a block and prints duration if verbosity is high enough.
#[macro_export]
macro_rules! cli_timed_v {
    ($verbosity:expr, $indent:expr, $msg:expr, $block:block) => {{
        if $crate::cli::printer_config().verbosity >= $verbosity {
            $crate::cli_timed!($indent, $msg, $block)
        } else {
            $block
        }
    }};
}

/// Helper function for macros to convert indent values to u32.
#[doc(hidden)]
#[inline(always)]
pub fn indent_as_u32<I: Into<u32>>(indent: I) -> u32 { indent.into() }

// Re-export macros at the cli module level
pub use crate::{cli_error, cli_note, cli_step, cli_success, cli_timed, cli_timed_v, cli_warning};
