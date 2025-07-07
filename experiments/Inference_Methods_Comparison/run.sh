#!/bin/bash

# Simple launcher script for inference comparison experiments

echo "🚀 Inference Methods Comparison"
echo "==============================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "inference_comparison.py" ]; then
    echo "❌ inference_comparison.py not found. Please run from Inference_Methods_Comparison directory."
    exit 1
fi

# Show usage options
echo "Available commands:"
echo "  ./run.sh test      - Test setup and dependencies"
echo "  ./run.sh quick     - Quick experiment (minimal settings)"
echo "  ./run.sh interactive - Interactive configuration"
echo "  ./run.sh launcher  - Interactive launcher menu"
echo "  ./run.sh help      - Show detailed help"
echo ""

# Handle command line arguments
case "${1:-launcher}" in
    "test")
        echo "🧪 Testing setup..."
        python3 test_setup.py
        ;;
    "quick")
        echo "⚡ Running quick experiment..."
        python3 inference_comparison.py --quick
        ;;
    "interactive")
        echo "🔧 Starting interactive mode..."
        python3 inference_comparison.py --interactive
        ;;
    "launcher")
        echo "🎯 Starting launcher..."
        python3 run_experiments.py
        ;;
    "help")
        echo "📚 Detailed help:"
        python3 inference_comparison.py --help
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Use: ./run.sh [test|quick|interactive|launcher|help]"
        exit 1
        ;;
esac
