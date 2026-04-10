# Script to setup Modal secrets for your MercuryMoE project
# This will create a secret named "mercury-moe-secrets" with your MODAL_API_KEY

echo "=== Modal Secret Setup ==="
echo ""
echo "This script will help you create a Modal secret with your API key."
echo ""
echo "IMPORTANT: You need to have already run 'modal token new' to authenticate."
echo ""

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "Modal CLI not found. Installing..."
    pip install modal
fi

# Check authentication
echo "Checking Modal authentication..."
modal token list 2>/dev/null
if [ $? -ne 0 ]; then
    echo ""
    echo "⚠️  You're not authenticated with Modal yet."
    echo "Please run: modal token new"
    echo "Then come back and run this script again."
    exit 1
fi

echo ""
echo "✓ Modal authentication confirmed"
echo ""

# Create the secret
echo "Creating Modal secret 'mercury-moe-secrets'..."
echo ""
echo "You will be prompted to enter your MODAL_API_KEY."
echo "Paste your API key when prompted (it won't be visible as you type):"
echo ""

# Use modal secret create with interactive input
modal secret create mercury-moe-secrets MODAL_API_KEY

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Secret 'mercury-moe-secrets' created successfully!"
    echo ""
    echo "Your modal_app.py is configured to use this secret."
    echo ""
    echo "Next steps:"
    echo "  1. Upload your data: python upload_data.py --data_dir <your_data_path>"
    echo "  2. Run training: modal run modal_app.py::train"
    echo "  3. Or get a shell: modal shell modal_app.py"
    echo ""
else
    echo ""
    echo "⚠️  Failed to create secret. Please try manually:"
    echo "   modal secret create mercury-moe-secrets MODAL_API_KEY=<your_key>"
    echo ""
fi

# Quick setup guide
echo "============================================"
echo "QUICK SETUP GUIDE"
echo "============================================"
echo ""
echo "Option 1: Interactive (recommended)"
echo "  modal secret create mercury-moe-secrets"
echo "  (Then follow the prompts to add MODAL_API_KEY)"
echo ""
echo "Option 2: One-line command"
echo "  modal secret create mercury-moe-secrets MODAL_API_KEY=<paste_your_key_here>"
echo ""
echo "Option 3: Via Modal Dashboard"
echo "  1. Go to https://modal.com/secrets"
echo "  2. Click 'Create Secret'"
echo "  3. Name it: mercury-moe-secrets"
echo "  4. Add key: MODAL_API_KEY"
echo "  5. Paste your API key value"
echo ""
echo "After creating the secret, your app will automatically use it!"
echo "============================================"
