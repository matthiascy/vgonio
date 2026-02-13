//! ANSI color codes and formatting utilities.

use std::fmt;

/// ANSI reset code.
pub const RESET: &str = "\u{001b}[0m";

/// A colored single-character symbol.
#[derive(Debug, Clone, Copy)]
pub struct ColoredSymbol {
    /// Symbol foreground color.
    pub color: Color,
    /// Symbol character.
    pub symbol: char,
}

impl ColoredSymbol {
    /// Creates a new colored symbol.
    pub const fn new(color: Color, symbol: char) -> Self { Self { color, symbol } }
}

impl fmt::Display for ColoredSymbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}{}", self.color.code(), self.symbol, RESET)
    }
}

/// Red exclamation marker.
pub const RED_EXCLAMATION: ColoredSymbol = ColoredSymbol::new(Color::Red, '!');
/// Cyan check marker.
pub const CYAN_CHECK: ColoredSymbol = ColoredSymbol::new(Color::Cyan, '✓');
/// Green check marker.
pub const GREEN_CHECK: ColoredSymbol = ColoredSymbol::new(Color::Green, '✓');
/// Yellow greater-than marker.
pub const YELLOW_GT: ColoredSymbol = ColoredSymbol::new(Color::Yellow, '>');
/// Cyan minus marker.
pub const CYAN_MINUS: ColoredSymbol = ColoredSymbol::new(Color::Cyan, '-');

/// ANSI color codes.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum Color {
    /// Cyan color.
    Cyan,
    /// Red color.
    Red,
    /// Green color.
    Green,
    /// Yellow color.
    Yellow,
}

impl Color {
    /// Returns the ANSI escape sequence for this color.
    pub const fn code(self) -> &'static str {
        match self {
            Color::Cyan => "\u{001b}[36m",
            Color::Red => "\u{001b}[31m",
            Color::Green => "\u{001b}[32m",
            Color::Yellow => "\u{001b}[33m",
        }
    }

    /// Wraps text with this color and reset.
    pub fn paint(self, text: &str) -> String { format!("{}{}{}", self.code(), text, RESET) }
}
