#!/usr/bin/env python
"""
Entry point for the CommentSense application.
This script runs the Streamlit frontend.
"""
import subprocess
import sys
import os

def main():
    """Run the Streamlit frontend."""
    # Path to the frontend main.py
    frontend_main = os.path.join(os.path.dirname(__file__), "src", "frontend", "main.py")
    
    # Run streamlit
    subprocess.run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        frontend_main
    ])

if __name__ == "__main__":
    main()
