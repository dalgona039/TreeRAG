
HOOKS_DIR=".git-hooks"
GIT_HOOKS_DIR=".git/hooks"

echo "🔧 Setting up Git hooks..."

if [ ! -d ".git" ]; then
    echo "❌ Error: Not a git repository"
    exit 1
fi

if [ -f "$HOOKS_DIR/pre-commit" ]; then
    cp "$HOOKS_DIR/pre-commit" "$GIT_HOOKS_DIR/pre-commit"
    chmod +x "$GIT_HOOKS_DIR/pre-commit"
    echo "✅ pre-commit hook installed"
else
    echo "⚠️  Warning: pre-commit hook not found in $HOOKS_DIR"
fi

echo ""
echo "Testing pre-commit hook..."
if [ -x "$GIT_HOOKS_DIR/pre-commit" ]; then
    echo "✅ Git hooks successfully installed and executable"
    echo ""
    echo "Next steps:"
    echo "  1. Try committing a file with 'AIzaSy' in it (will be blocked)"
    echo "  2. Check SECURITY.md for best practices"
else
    echo "❌ Error: Hook installation failed"
    exit 1
fi
