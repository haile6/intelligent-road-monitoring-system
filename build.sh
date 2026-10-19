#!/usr/bin/env bash
# Render Build Script for Intelligent Road Monitoring System
set -o errexit

echo "[*] Upgrading pip..."
pip install --upgrade pip

echo "[*] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "[*] Initializing SQLite database schema and pre-seeding records..."
python -c "from database import db_manager; db_manager.init_db(); print('[+] Database successfully verified on cloud build.')"

echo "[*] Build completed successfully."
