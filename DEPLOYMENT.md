# UCLA CS Server Deployment Guide

This guide will help you deploy the CS145 Leaderboard to your UCLA CS personal web space at `https://web.cs.ucla.edu/~fts/`.

## 🎯 Overview

The deployment will create a publicly accessible leaderboard at:
```
https://web.cs.ucla.edu/~fts/cs145-leaderboard/
```

## 📋 Prerequisites

1. **SSH Access**: Ensure you can SSH to `lion.cs.ucla.edu`
2. **Web Space**: Confirm your `~/www` directory exists on the UCLA CS servers
3. **Java**: Required for PySpark (usually pre-installed on CS servers)
4. **Course Files**: Make sure you have all required files in your project directory

## 🚀 Quick Deployment

### Step 1: Make the deployment script executable
```bash
chmod +x deploy.sh
```

### Step 2: Run the deployment
```bash
./deploy.sh
```

The script will automatically:
- Copy all necessary files to `~/www/cs145-leaderboard/`
- Set proper permissions
- Install Python dependencies
- Test the deployment

### Step 3: Verify deployment
Visit: `https://web.cs.ucla.edu/~fts/cs145-leaderboard/`

## 🔧 Manual Deployment (Alternative)

If the automated script doesn't work, follow these manual steps:

### 1. SSH to UCLA CS servers
```bash
ssh fts@lion.cs.ucla.edu
```

### 2. Create directory structure
```bash
mkdir -p ~/www/cs145-leaderboard/{templates,uploads,static}
cd ~/www/cs145-leaderboard
```

### 3. Copy files from your local machine
From your local project directory:
```bash
# Copy main application files
scp app_production.py fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/
scp wsgi.py fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/
scp evaluation.py fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/
scp .htaccess fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/
scp requirements.txt fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/

# Copy templates
scp templates/*.html fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/templates/

# Copy course files
scp data_generator.py simulator.py config.py sample_recommenders.py fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/

# Copy sim4rec directory
scp -r sim4rec fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/
```

### 4. Set permissions on the server
```bash
ssh fts@lion.cs.ucla.edu
cd ~/www/cs145-leaderboard
chmod 755 .
chmod 644 *.py
chmod 755 wsgi.py
chmod 644 templates/*.html
chmod 644 .htaccess
chmod 755 uploads
chmod -R 755 sim4rec
```

### 5. Install dependencies
```bash
python3 -m pip install --user -r requirements.txt
```

## 🔍 Testing & Troubleshooting

### Test the deployment
```bash
# SSH to the server
ssh fts@lion.cs.ucla.edu
cd ~/www/cs145-leaderboard

# Test Python imports
python3 -c "from app_production import app; print('✅ App imports successfully')"

# Check file structure
ls -la
```

### Common Issues & Solutions

#### 1. **Module Not Found Errors**
```bash
# Check if all files are present
ls -la ~/www/cs145-leaderboard/
ls -la ~/www/cs145-leaderboard/sim4rec/

# Reinstall dependencies
python3 -m pip install --user -r requirements.txt
```

#### 2. **Permission Denied**
```bash
# Fix permissions
chmod -R 755 ~/www/cs145-leaderboard/
chmod 644 ~/www/cs145-leaderboard/*.py
chmod 755 ~/www/cs145-leaderboard/wsgi.py
```

#### 3. **Apache Errors**
Check the error log:
```bash
tail -f ~/www/cs145-leaderboard/error.log
```

#### 4. **Database Issues**
The SQLite database will be created automatically. If you encounter issues:
```bash
# Remove and recreate database
rm ~/www/cs145-leaderboard/leaderboard.db
# The app will recreate it automatically
```

### Debugging Commands

```bash
# Check Apache configuration
cat ~/www/cs145-leaderboard/.htaccess

# Check Python path and modules
python3 -c "import sys; print('\n'.join(sys.path))"

# Test evaluation script
cd ~/www/cs145-leaderboard
python3 -c "from evaluation import run_evaluation; print('Evaluation module OK')"

# Check uploads directory
ls -la uploads/
```

## 🔄 Updates & Maintenance

### To update the application
1. Make changes locally
2. Run `./deploy.sh` again
3. The script will overwrite files with the latest version

### To monitor submissions
- Visit: `https://web.cs.ucla.edu/~fts/cs145-leaderboard/admin`
- Check logs: `ssh fts@lion.cs.ucla.edu 'tail -f ~/www/cs145-leaderboard/error.log'`

### To backup data
```bash
# Backup database
scp fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/leaderboard.db ./backup_$(date +%Y%m%d).db

# Backup uploads
scp -r fts@lion.cs.ucla.edu:~/www/cs145-leaderboard/uploads ./uploads_backup_$(date +%Y%m%d)/
```

## 🌐 Access URLs

- **Main Page**: https://web.cs.ucla.edu/~fts/cs145-leaderboard/
- **Leaderboard**: https://web.cs.ucla.edu/~fts/cs145-leaderboard/leaderboard
- **Admin Dashboard**: https://web.cs.ucla.edu/~fts/cs145-leaderboard/admin
- **API**: https://web.cs.ucla.edu/~fts/cs145-leaderboard/api/leaderboard

## 📧 Support

If you encounter issues:
1. Check this troubleshooting guide
2. Review the error logs
3. Test individual components manually
4. Contact UCLA CS support for server-specific issues

---

**Note**: The UCLA CS servers use Apache with mod_wsgi to serve Python applications. The `.htaccess` file configures this setup automatically. 