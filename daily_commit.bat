@echo off
REM Daily commit script for SupplySync AI
cd /d "%~dp0"

REM Add all changes
git add .

REM Commit with timestamp
git commit -m "Daily update: %date% %time%"

REM Push to GitHub
git push

echo Daily commit completed!
