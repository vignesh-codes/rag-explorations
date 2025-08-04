#!/usr/bin/env python3
"""Setup script for RAG Explorations project."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def setup_environment():
    """Set up environment file."""
    if not Path('.env').exists():
        if Path('.env.example').exists():
            shutil.copy('.env.example', '.env')
            print("✅ Created .env file from example")
            print("📝 Please edit .env with your API keys")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("ℹ️  .env file already exists")
    return True


def setup_data_directory():
    """Set up data directory and move PDF if needed."""
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    pdf_file = Path('coding_interview_book.pdf')
    target_pdf = data_dir / 'coding_interview_book.pdf'
    
    if pdf_file.exists() and not target_pdf.exists():
        shutil.move(str(pdf_file), str(target_pdf))
        print("✅ Moved PDF file to data directory")
    elif target_pdf.exists():
        print("ℹ️  PDF file already in data directory")
    else:
        print("ℹ️  No PDF file found to move")
    
    return True


def install_dependencies():
    """Install project dependencies."""
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def install_dev_dependencies():
    """Install development dependencies."""
    return run_command("pip install -e .[dev]", "Installing development dependencies")


def run_tests():
    """Run the test suite."""
    return run_command("python -m pytest", "Running tests")


def format_code():
    """Format code with black and isort."""
    success = True
    success &= run_command("python -m black src tests cli.py", "Formatting with black")
    success &= run_command("python -m isort src tests cli.py", "Sorting imports with isort")
    return success


def lint_code():
    """Run linting checks."""
    success = True
    success &= run_command("python -m flake8 src tests cli.py", "Running flake8")
    success &= run_command("python -m mypy src", "Running mypy type checking")
    return success


def main():
    """Main setup function."""
    print("🚀 Setting up RAG Explorations project...")
    
    # Basic setup
    setup_environment()
    setup_data_directory()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "install":
            install_dependencies()
        elif command == "install-dev":
            install_dev_dependencies()
        elif command == "test":
            run_tests()
        elif command == "format":
            format_code()
        elif command == "lint":
            lint_code()
        elif command == "all":
            install_dependencies()
            run_tests()
            print("\n🎉 Setup completed successfully!")
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: install, install-dev, test, format, lint, all")
            sys.exit(1)
    else:
        print("\n📋 Available commands:")
        print("  python setup.py install      - Install dependencies")
        print("  python setup.py install-dev  - Install dev dependencies")
        print("  python setup.py test         - Run tests")
        print("  python setup.py format       - Format code")
        print("  python setup.py lint         - Run linting")
        print("  python setup.py all          - Full setup")
        print("\n💡 Run 'python setup.py all' for complete setup")


if __name__ == "__main__":
    main()