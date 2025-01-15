mod asset;
pub use asset::Asset;
pub mod handle;
#[cfg(feature = "winit")]
pub mod input;
pub mod medium;
pub mod partition;
pub mod range;

use chrono::{DateTime, Local};

/// Returns the current time as an ISO 8601 (RFC 3339) timestamp.
pub fn iso_timestamp() -> String {
    chrono::Local::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
}

/// Returns the current time as an ISO 8601 (RFC 3339) timestamp without the
/// timezone and the colon in the time field.
pub fn iso_timestamp_short(datetime: DateTime<Local>) -> String {
    datetime.format("%Y-%m-%dT%H-%M-%S").to_string()
}

/// Converts a date time to an ISO 8601 (RFC 3339) timestamp.
pub fn iso_timestamp_from_datetime(dt: &chrono::DateTime<chrono::Local>) -> String {
    dt.to_rfc3339_opts(chrono::SecondsFormat::Micros, false)
}

/// Converts a date time to an ISO 8601 (RFC 3339) timestamp without the
/// timezone and with the colon in the time field.
pub fn iso_timestamp_display(dt: &chrono::DateTime<chrono::Local>) -> String {
    dt.format("%Y-%m-%d %H:%M:%S").to_string()
}
