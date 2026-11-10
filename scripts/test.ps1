# CI-style: run all tests and write dated evidence.  Usage (PowerShell): .\scripts\test.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
python scripts\record_evidence.py
exit $LASTEXITCODE
