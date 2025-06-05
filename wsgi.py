#!/usr/bin/env python3
import sys
import os

# Add the application directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import from production app
from app_production import app as application

if __name__ == "__main__":
    application.run() 