#!/usr/bin/env sh

# The script is intended to be run from the root directory of the project, it contains the commands to
#
# 1. setup the development environment
# 2. facilitate the development process
# 3. run the predefined commands concerning the surface related measurements

help() {
    echo "Usage: $0 {pybind|} [options]"
    echo
    echo "Commands:"
    echo "  pybind          Build the python module."
    echo "  setup           Set up the development environment."
    echo "  help            Show this help message."
    echo
    echo "Run '$0 <command> --help' for more information on a command."
    echo
}

setup_env() {
    echo "Setting up the development environment..."
    setup_python
}

check_install_python_packages() {
  # Check if the required python packages are installed as specified in the requirements.txt file
  echo "Checking if the required python packages are installed..."
  while IFS= read -r line || [ -n "$line" ]; do
    # Extract the package name from the line
    pkg_name=$(echo "$line" | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1)
    # Remove potential ~ from the package name
    pkg_name=$(echo "$pkg_name" | tr -d '~')
    # print without newline
    echo -n "  -- checking [$pkg_name] "

    if pip show "$pkg_name" > /dev/null 2>&1; then
      # print a checkmark at the end of the line
      echo "✓"
    else
      echo "The python package [$pkg_name] is not installed. Installing..."
        pip install "$pkg_name"
        if [ $? -eq 0 ]; then
          echo "The python package [$pkg_name] has been installed."
        else
          echo "Error: failed to install the python package [$pkg_name]."
          exit 1
        fi
    fi
  done < requirements.txt
  echo
}

setup_python() {
    echo "Setting up the python virtual environment..."
    echo
    echo "Checking if the python virtual environment exists..."
    if [ ! -d ".pyenv" ]; then
        echo "Creating the python virtual environment..."
        # Test if the python3 command is available
        if ! command -v python3 > /dev/null 2>&1; then
            echo "Error: python3 is not installed."
            exit 1
        fi
        # Create the python virtual environment
        python3 -m venv .pyenv
        echo "The python virtual environment has been created."
    else
        echo "The python virtual environment already exists."
    fi
    echo

    # Activate the python virtual environment if it is not activated
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "Activating the python virtual environment..."
        . .pyenv/bin/activate
        echo "The python virtual environment has been activated."
    else
        echo "The python virtual environment is already activated."
    fi
    echo

    check_install_python_packages

    echo "The development environment has been set up."
    echo
}

build_python_module() {
    PKG_NAME=""
    while [ "$#" -gt 0 ]; do
        case "$1" in
            -n|--name)
                PKG_NAME="$2"
                shift 2
                ;;
            -r|-rel|--release)
                RELEASE=true
                shift
                ;;
            -d|-dev|--develop)
                DEVELOP=true
                shift
                ;;
            -h|--help)
                help_build_python_module
                exit 0
                ;;
            *)
                echo "Error: unknown option '$1'"
                help
                exit 1
                ;;
        esac
    done

    if [ -z "$PKG_NAME" ]; then
        echo "Error: missing required option '-n'"
        help_build_python_module
        exit 1
    fi

    echo "Building the python module ['$PKG_NAME']..."

    setup_python

    if [ -n "$RELEASE" ]; then
      echo "Building the python wheels..."
      # Build the wheels and stores them in a folder (target/wheels by default.
      maturin build -m crates/$PKG_NAME/Cargo.toml --release --features pybind
      return
    elif [ -n "$DEVELOP" ]; then
      echo "Building the python module into the virtual environment..."
      # Build the crate and installs it as a python module directly in the virtual environment
      maturin develop -m crates/$PKG_NAME/Cargo.toml --release --features pybind
      return
    else
      echo "No build option specified. Please specify either --release or --develop."
    fi
}

help_build_python_module() {
    echo "Usage: $0 pybind [options]"
    echo
    echo "Options:"
    echo "  -n, --name      The name of the python package to build."
    echo "  -h, --help      Show this help message."
}

if [ $# -eq 0 ]; then
    help
    exit 1
fi

# Make sure the script is run from the root directory of the project
if [ ! -f "vgdev.sh" ]; then
    echo "Error: the script must be run from the root directory of the project."
    exit 1
fi

# Parse the command, and shift it from the arguments
CMD="$1"
shift

# Execute the command
case "$CMD" in
    pybind)
        build_python_module "$@"
        ;;
    setup)
        setup_env
        ;;
    -h|--help|help)
        help
        ;;
    *)
        echo "Error: unknown command '$CMD'"
        help
        exit 1
        ;;
esac