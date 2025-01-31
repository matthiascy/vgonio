use log::LevelFilter;

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

/// A filter for the logger.
///
/// This is a tuple of a module name and a log level filter.
pub type LogFilter<'a> = (&'a str, LevelFilter);

/// Initialises logging settings.
///
/// # Arguments
///
/// * `timestamp` - Whether to print the timestamp in the log; This is the base
///   time for the timestamp.
/// * `log_level` - The log level to filter. This is the top level log level for
///   the program. See [`log_filter_from_level`] for more details.
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
