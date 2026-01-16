/// ANSI color codes.
pub const BRIGHT_CYAN: &str = "\u{001b}[36m";
pub const BRIGHT_RED: &str = "\u{001b}[31m";
pub const BRIGHT_GREEN: &str = "\u{001b}[32m";
pub const BRIGHT_YELLOW: &str = "\u{001b}[33m";
pub const RESET: &str = "\u{001b}[0m";

pub const RED_EXCLAMATION: &str = "\u{001b}[31m!\u{001b}[0m";
pub const CYAN_CHECK: &str = "\u{001b}[36m✓\u{001b}[0m";
pub const GREEN_CHECK: &str = "\u{001b}[32m✓\u{001b}[0m";

pub const YELLOW_GT: &str = "\u{001b}[33m>\u{001b}[0m";

pub const CYAN_MINUS: &str = "\u{001b}[36m-\u{001b}[0m";

/// ANSI color codes.
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum Color {
    BrightCyan = 36,
    BrightRed = 31,
    BrightGreen = 32,
    BrightYellow = 33,
}
