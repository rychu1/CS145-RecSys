#!/bin/bash

# Deployment script for CS145 Leaderboard to UCLA CS servers
# Usage: ./deploy.sh

set -e

# Configuration
REMOTE_USER="fts"
REMOTE_HOST="lion.cs.ucla.edu"
REMOTE_PATH="~/www/cs145-leaderboard"
LOCAL_PATH="."

echo "🚀 Deploying CS145 Leaderboard to UCLA CS servers..."

# Create remote directory structure
echo "📁 Creating remote directory structure..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}/{templates,uploads,static}"

# Copy application files
echo "📤 Copying application files..."
scp app_production.py ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
scp wsgi.py ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
scp evaluation.py ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
scp requirements.txt ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
scp README.md ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

# Copy templates
echo "📄 Copying templates..."
scp templates/*.html ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/templates/

# Copy course-specific files
echo "📚 Copying course files..."
if [ -f "data_generator.py" ]; then
    scp data_generator.py ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
fi
if [ -f "simulator.py" ]; then
    scp simulator.py ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
fi
if [ -f "config.py" ]; then
    scp config.py ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
fi
if [ -f "sample_recommenders.py" ]; then
    scp sample_recommenders.py ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
fi

# Copy sim4rec directory
echo "🧠 Copying sim4rec module..."
if [ -d "sim4rec" ]; then
    scp -r sim4rec ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/
fi

# Copy Apache configuration
echo "⚙️  Copying Apache configuration..."
scp .htaccess ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/

# Set permissions
echo "🔐 Setting permissions..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "
    cd ${REMOTE_PATH}
    chmod 755 .
    chmod 644 *.py
    chmod 755 wsgi.py
    chmod 644 templates/*.html
    chmod 644 .htaccess
    chmod 755 uploads
    chmod 644 *.txt *.md
    # Set proper permissions for sim4rec directory
    if [ -d sim4rec ]; then
        chmod -R 755 sim4rec
        find sim4rec -name '*.py' -exec chmod 644 {} \;
    fi
"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "
    cd ${REMOTE_PATH}
    python3 -m pip install --user -r requirements.txt
"

# Test the deployment
echo "🧪 Testing deployment..."
ssh ${REMOTE_USER}@${REMOTE_HOST} "
    cd ${REMOTE_PATH}
    python3 -c 'from app_production import app; print(\"✅ App imports successfully\")'
"

echo "✅ Deployment completed!"
echo "🌐 Your leaderboard should be accessible at:"
echo "   https://web.cs.ucla.edu/~fts/cs145-leaderboard/"
echo ""
echo "📝 Next steps:"
echo "   1. Test the deployment by visiting the URL above"
echo "   2. Check Apache error logs if there are issues:"
echo "      ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ~/www/cs145-leaderboard/error.log'"
echo "   3. Monitor application logs in the uploads directory"
echo "   4. You can check the status with:"
echo "      ssh ${REMOTE_USER}@${REMOTE_HOST} 'ls -la ~/www/cs145-leaderboard/'" 