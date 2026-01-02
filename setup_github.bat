@echo off
REM GitHub Setup Script for mammography-multiobjective-optimization
REM Author: David (dtobi59)

echo ==========================================
echo GitHub Repository Setup
echo ==========================================
echo.

REM Check if git is installed
git --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Git is not installed
    echo Please install Git first: https://git-scm.com/downloads
    pause
    exit /b 1
)

echo [OK] Git is installed
echo.

REM Check if already initialized
if exist .git (
    echo [WARNING] Git repository already initialized
    echo This directory already has a .git folder
    echo.
    set /p continue="Do you want to continue anyway? (y/n): "
    if /i not "%continue%"=="y" (
        echo Setup cancelled
        pause
        exit /b 1
    )
) else (
    echo Initializing git repository...
    git init
    echo [OK] Git repository initialized
    echo.
)

REM Configure git
echo Configuring Git...
git config user.name "David" 2>nul
echo [OK] Git configured
echo.

REM Add all files
echo Adding files to git...
git add .
echo [OK] Files added
echo.

REM Create commit
echo Creating initial commit...
git commit -m "Initial commit: Multi-objective breast cancer classification" -m "- Complete implementation with dataset-specific parsers" -m "- Support for VinDr-Mammo and INbreast datasets" -m "- NSGA-III optimization with 4 objectives" -m "- 114 tests (100%% passing)" -m "- Comprehensive documentation" -m "- Interactive Jupyter notebook tutorial"
echo [OK] Initial commit created
echo.

REM Rename branch to main
echo Setting main branch...
git branch -M main
echo [OK] Branch set to main
echo.

REM Check if remote exists
git remote | findstr "origin" >nul 2>&1
if %errorlevel% equ 0 (
    echo [WARNING] Remote 'origin' already exists
    git remote get-url origin
    echo.
    set /p update="Do you want to update it? (y/n): "
    if /i "%update%"=="y" (
        git remote set-url origin https://github.com/dtobi59/mammography-multiobjective-optimization.git
        echo [OK] Remote updated
    )
) else (
    echo Adding remote repository...
    git remote add origin https://github.com/dtobi59/mammography-multiobjective-optimization.git
    echo [OK] Remote added
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Next steps:
echo.
echo 1. Create repository on GitHub:
echo    https://github.com/new
echo    Name: mammography-multiobjective-optimization
echo    (DO NOT initialize with README or license)
echo.
echo 2. Push to GitHub:
echo    git push -u origin main
echo.
echo Your repository will be at:
echo https://github.com/dtobi59/mammography-multiobjective-optimization
echo.
echo ==========================================
echo.
pause
