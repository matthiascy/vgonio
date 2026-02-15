use vgn_core::{
    cli::{self, setup_printer, ColorMode, Indent, PrinterConfig},
    cli_error, cli_note, cli_note_v, cli_step, cli_step_v, cli_success, cli_timed, cli_warning,
};

#[test]
fn test_cli_utilities() {
    use std::time::Duration;

    // Test duration formatting
    assert_eq!(cli::format_duration(Duration::from_millis(500)), "500ms");
    assert_eq!(cli::format_duration(Duration::from_millis(1500)), "1.500s");
    assert_eq!(cli::format_duration(Duration::from_secs(75)), "1m 15s");
    assert_eq!(cli::format_duration(Duration::from_secs(3661)), "1h 1m 1s");
    assert!(cli::format_duration(Duration::from_micros(500)).contains("s")); // Accept either μ or µ
    assert_eq!(cli::format_duration(Duration::from_nanos(500)), "500ns");

    // Test duration seconds formatting
    assert_eq!(
        cli::format_duration_secs(Duration::from_millis(1500), 2),
        "1.50s"
    );
    assert_eq!(cli::format_duration_secs(Duration::from_secs(42), 0), "42s");

    // Test byte formatting
    assert_eq!(cli::format_bytes(512), "512 B");
    assert_eq!(cli::format_bytes(2048), "2.00 KiB");
    assert_eq!(cli::format_bytes(1_048_576), "1.00 MiB");
    assert_eq!(cli::format_bytes(1_073_741_824), "1.00 GiB");

    // Test Indent Display impl
    assert_eq!(format!("{}", Indent::ROOT), "0sp");
    assert_eq!(format!("{}", Indent::SECTION), "2sp");
    assert_eq!(format!("{}", Indent::DETAIL), "6sp");

    // Test Indent Default impl
    assert_eq!(Indent::default(), Indent::ROOT);

    // Test Indent nesting
    let indent = Indent::SECTION.nest();
    assert_eq!(indent, Indent::SUBSECTION);
}

#[test]
fn test_cli_print() {
    // Setup printer with verbosity level 1
    setup_printer(PrinterConfig {
        quiet: false,
        verbosity: 1,
        color_mode: ColorMode::Auto,
    });

    println!("=== Testing CLI ===\n");

    // Test 1: Ergonomic macros
    println!("Test 1: Ergonomic macros");
    cli_step!(Indent::ROOT, "Starting process...");
    cli_note!(Indent::SECTION, "Processing {} items", 42);
    cli_success!(Indent::SECTION, "Completed successfully");
    cli_warning!(Indent::SECTION, "Deprecated feature detected");
    cli_error!(Indent::ROOT, "Critical error occurred");

    // Test 2: Type-safe indentation
    println!("\nTest 2: Type-safe indentation");
    let root = Indent::ROOT;
    let section = Indent::SECTION;
    let nested = section.nest();
    cli_step!(root, "Root level ({})", root.as_u32());
    cli_step!(section, "Section level ({})", section.as_u32());
    cli_step!(nested, "Nested level ({})", nested.as_u32());
    cli_step!(Indent::DETAIL, "Detail level");
    cli_step!(Indent::DEEP, "Deep level");

    // Test 3: Legacy constants
    println!("\nTest 3: Legacy indent constants");
    cli::step(
        Indent::ROOT.into(),
        format_args!("Using legacy ROOT constant"),
    );
    cli::step(
        Indent::SECTION.into(),
        format_args!("Using legacy SECTION constant"),
    );

    // Test 4: Zero-cost verbosity gating
    println!("\nTest 4: Zero-cost verbosity gating (verbosity = 1)");
    cli_step_v!(0, Indent::ROOT, "This shows (requires verbosity >= 0)");
    cli_step_v!(
        1,
        Indent::SECTION,
        "This also shows (requires verbosity >= 1)"
    );
    cli_step_v!(
        2,
        Indent::SECTION,
        "This is hidden (requires verbosity >= 2)"
    );

    // Test 5: Mixed u32 and Indent types
    println!("\nTest 5: Mixed indentation types");
    cli_step!(0u32, "Using u32 directly");
    cli_step!(Indent::SECTION, "Using Indent type");
    cli_step!(Indent::custom(10), "Using custom indent (10 spaces)");

    // Test 6: Warning function (both function and macro)
    println!("\nTest 6: Warning output");
    cli::warning(0, format_args!("Warning via function"));
    cli_warning!(2u32, "Warning via macro");
    cli_note_v!(1, 4u32, "Note with verbosity check (shown)");

    // Test 7: Timed operations
    println!("\nTest 7: Timed operations");
    let result = cli_timed!(Indent::ROOT, "Simulating work", {
        std::thread::sleep(std::time::Duration::from_millis(50));
        42
    });
    assert_eq!(result, 42);

    println!("\n=== All tests completed ===");
}
