#!/bin/bash

# GitHub Setup Script for mammography-multiobjective-optimization
# Author: David (dtobi59)

echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Error: Git is not installed"
    echo "Please install Git first: https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git is installed"
echo ""

# Check if already initialized
if [ -d .git ]; then
    echo "⚠️  Git repository already initialized"
    echo "This directory already has a .git folder"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Setup cancelled"
        exit 1
    fi
else
    echo "Initializing git repository..."
    git init
    echo "✅ Git repository initialized"
    echo ""
fi

# Configure git (optional)
echo "Configuring Git..."
git config user.name "David" 2>/dev/null || true
echo "✅ Git configured"
echo ""

# Add all files
echo "Adding files to git..."
git add .
echo "✅ Files added"
echo ""

# Create commit
echo "Creating initial commit..."
git commit -m "Initial commit: Multi-objective breast cancer classification

- Complete implementation with dataset-specific parsers
- Support for VinDr-Mammo and INbreast datasets
- NSGA-III optimization with 4 objectives and checkpointing
- Google Colab integration with one-click setup
- 118 tests (100% passing)
- Comprehensive documentation
- Interactive tutorials (local + Colab)"

echo "✅ Initial commit created"
echo ""

# Rename branch to main
echo "Setting main branch..."
git branch -M main
echo "✅ Branch set to main"
echo ""

# Check if remote already exists
if git remote | grep -q origin; then
    echo "⚠️  Remote 'origin' already exists"
    echo "Current remote URL:"
    git remote get-url origin
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote set-url origin https://github.com/dtobi59/mammography-multiobjective-optimization.git
        echo "✅ Remote updated"
    fi
else
    echo "Adding remote repository..."
    git remote add origin https://github.com/dtobi59/mammography-multiobjective-optimization.git
    echo "✅ Remote added"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Create repository on GitHub:"
echo "   https://github.com/new"
echo "   Name: mammography-multiobjective-optimization"
echo "   (DO NOT initialize with README or license)"
echo ""
echo "2. Push to GitHub:"
echo "   git push -u origin main"
echo ""
echo "Your repository will be at:"
echo "https://github.com/dtobi59/mammography-multiobjective-optimization"
echo ""
echo "=========================================="
