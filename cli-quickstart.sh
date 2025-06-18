#!/bin/bash

# IQID-Alphas CLI Quick Reference
# ================================

echo "🔬 IQID-Alphas Advanced Batch Processing CLI"
echo "============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "iqid-cli.py" ]; then
    echo "❌ Error: Please run this from the iqid-alphas directory"
    exit 1
fi

echo "📋 Quick Commands:"
echo ""

echo "1. Get Help:"
echo "   python -m iqid_alphas.cli --help"
echo "   ./iqid-cli.py --help"
echo ""

echo "2. Discover Data:"
echo "   python -m iqid_alphas.cli discover --data data/"
echo "   python -m iqid_alphas.cli discover --data data/ --output discovery.json"
echo ""

echo "3. Create Configuration:"
echo "   python -m iqid_alphas.cli config --create configs/my_config.json"
echo "   python -m iqid_alphas.cli config --validate configs/my_config.json"
echo ""

echo "4. Process Samples:"
echo "   # Quick test (3 samples)"
echo "   python -m iqid_alphas.cli process --data data/ --config configs/cli_quick_config.json --max-samples 3"
echo ""
echo "   # Full batch processing"
echo "   python -m iqid_alphas.cli process --data data/ --config configs/cli_batch_config.json --pipeline advanced"
echo ""
echo "   # With custom output directory"
echo "   python -m iqid_alphas.cli process --data data/ --config configs/cli_batch_config.json --output results/my_batch"
echo ""

echo "🔧 Available Pipelines:"
echo "   - simple    : Basic iQID processing"
echo "   - advanced  : Comprehensive analysis with quality metrics"
echo "   - combined  : Joint iQID + H&E processing"
echo ""

echo "📁 Pre-configured Files:"
echo "   - configs/cli_batch_config.json  : Full-featured batch processing"
echo "   - configs/cli_quick_config.json  : Fast processing for testing"
echo ""

echo "📖 Documentation:"
echo "   - docs/user_guides/cli_guide.md  : Complete CLI documentation"
echo "   - CLI_IMPLEMENTATION_COMPLETE.md : Implementation overview"
echo ""

echo "🧪 Test the CLI:"
echo "   ./test_cli.py  : Run CLI test suite"
echo ""

echo "🎯 Example Workflow:"
echo "   1. python -m iqid_alphas.cli discover --data data/"
echo "   2. python -m iqid_alphas.cli config --create configs/test.json"
echo "   3. python -m iqid_alphas.cli process --data data/ --config configs/test.json --max-samples 5"
echo ""

echo "✅ CLI is ready to use! Choose a command above to get started."
