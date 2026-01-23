//! ANSI color codes.

/// Bright cyan ANSI color code.
pub const BRIGHT_CYAN: &str = "\u{001b}[36m";

/// Bright red ANSI color code.
pub const BRIGHT_RED: &str = "\u{001b}[31m";

/// Bright green ANSI color code.
pub const BRIGHT_GREEN: &str = "\u{001b}[32m";

/// Bright yellow ANSI color code.
pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";

/// ANSI reset code.
pub const RESET: &str = "\u{001b}[0m";

/// Colored symbols.
pub const RED_EXCLAMATION: &str = "\u{001b}[31m!\u{001b}[0m";

/// Colored check marks.
pub const CYAN_CHECK: &str = "\u{001b}[36m✓\u{001b}[0m";

/// Colored check marks.
pub const GREEN_CHECK: &str = "\u{001b}[32m✓\u{001b}[0m";

/// Colored greater-than signs.
pub const YELLOW_GT: &str = "\u{001b}[33m>\u{001b}[0m";

/// Colored minus signs.
pub const CYAN_MINUS: &str = "\u{001b}[36m-\u{001b}[0m";

/// ANSI color codes.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Color {
    /// Bright cyan.
    BrightCyan = 36,
    /// Bright red.
    BrightRed = 31,
    /// Bright green.
    BrightGreen = 32,
    /// Bright yellow.
    BrightYellow = 33,
}
