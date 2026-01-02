# GitHub Setup Instructions

Follow these steps to push this project to GitHub.

## Prerequisites

1. **Git installed**: Check with `git --version`
2. **GitHub account**: Logged in as **dtobi59**
3. **Git configured**:
   ```bash
   git config --global user.name "David"
   git config --global user.email "your-email@example.com"
   ```

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in:
   - **Repository name:** `mammography-multiobjective-optimization`
   - **Description:** Multi-objective hyperparameter optimization for breast cancer classification under dataset shift
   - **Visibility:** Public (or Private if you prefer)
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
3. Click "Create repository"

## Step 2: Initialize Local Repository

Open terminal/command prompt in the project directory:

```bash
cd C:\Users\HP\Downloads\project
```

Then run these commands:

```bash
# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Multi-objective breast cancer classification

- Complete implementation with dataset-specific parsers
- Support for VinDr-Mammo and INbreast datasets
- NSGA-III optimization with 4 objectives and checkpointing
- Google Colab integration with one-click setup
- 118 tests (100% passing)
- Comprehensive documentation
- Interactive tutorials (local + Colab)"

# Rename branch to main (if needed)
git branch -M main

# Add remote repository
git remote add origin https://github.com/dtobi59/mammography-multiobjective-optimization.git

# Push to GitHub
git push -u origin main
```

## Step 3: Verify Upload

1. Go to https://github.com/dtobi59/mammography-multiobjective-optimization
2. Verify all files are uploaded
3. Check that README.md displays correctly
4. Verify badges show up (tests may show "no status" until first run)

## Step 4: Set Up GitHub Actions (Optional)

GitHub Actions will automatically run tests on every push.

1. Go to repository → Actions tab
2. You should see the "Tests" workflow
3. It will run automatically on the next push

To trigger it manually:
```bash
# Make a small change
echo "" >> README.md

# Commit and push
git add .
git commit -m "Trigger GitHub Actions"
git push
```

## Step 5: Add Topics (Recommended)

On your repository page:
1. Click the gear icon (⚙️) next to "About"
2. Add topics:
   - `breast-cancer`
   - `mammography`
   - `deep-learning`
   - `multi-objective-optimization`
   - `nsga3`
   - `pytorch`
   - `medical-imaging`
   - `domain-shift`
   - `resnet`
   - `zero-shot-learning`
3. Add a description (optional)
4. Add website (optional)
5. Save changes

## Step 6: Create Release (Optional)

1. Go to repository → Releases → "Create a new release"
2. Tag: `v1.0.0`
3. Title: `v1.0.0 - Initial Release`
4. Description:
   ```
   Initial release of multi-objective hyperparameter optimization for breast cancer classification.

   ## Features
   - Dataset-specific parsers for VinDr-Mammo and INbreast
   - BI-RADS to binary label mapping (including subcategories)
   - ResNet-50 with partial fine-tuning
   - NSGA-III optimization (4 objectives) with automatic checkpointing
   - Google Colab integration - run with one click, no setup required
   - Zero-shot transfer evaluation
   - 118 tests (100% passing)
   - Comprehensive documentation
   - Interactive tutorials (local + Colab)

   ## Datasets Supported
   - VinDr-Mammo (source/training)
   - INbreast (target/zero-shot evaluation)
   - Demo synthetic dataset for quick testing
   ```
5. Publish release

## Troubleshooting

### Authentication Issues

If you get authentication errors:

**Option 1: Personal Access Token (Recommended)**
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token (classic)
3. Select scopes: `repo` (all)
4. Copy token
5. Use token as password when pushing

**Option 2: SSH**
1. Generate SSH key: `ssh-keygen -t ed25519 -C "your-email@example.com"`
2. Add to GitHub: Settings → SSH and GPG keys → New SSH key
3. Change remote URL:
   ```bash
   git remote set-url origin git@github.com:dtobi59/mammography-multiobjective-optimization.git
   ```

### Large Files

If you get "file too large" errors:
1. Check `.gitignore` is working
2. Remove large files from git:
   ```bash
   git rm --cached path/to/large/file
   git commit -m "Remove large file"
   ```

### Wrong Remote

If you need to change the remote URL:
```bash
git remote set-url origin https://github.com/dtobi59/mammography-multiobjective-optimization.git
```

## Quick Reference Commands

```bash
# Check status
git status

# Add files
git add .

# Commit changes
git commit -m "Your message"

# Push to GitHub
git push

# Pull from GitHub
git pull

# View remotes
git remote -v

# View commit history
git log --oneline
```

## Next Steps After Upload

1. ✅ Repository is live
2. Star your own repository (optional)
3. Watch the repository for updates
4. Share the link!
5. Consider adding:
   - GitHub Pages for documentation
   - More examples
   - Pre-trained models (if sharing)

## Repository URL

Once created, your repository will be at:

**https://github.com/dtobi59/mammography-multiobjective-optimization**

Share this link in papers, presentations, or with collaborators!

## Questions?

- Check Git documentation: https://git-scm.com/doc
- GitHub help: https://docs.github.com
- Open an issue if you encounter problems

---

**Ready to go!** Just follow the commands in Step 2 above.
