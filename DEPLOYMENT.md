# 🚀 CI/CD Deployment Guide for Customer Churn Prediction

This guide will help you set up automated deployment for your Customer Churn Prediction application.

## 📋 Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [GitHub Secrets Setup](#github-secrets-setup)
4. [Deployment Platforms](#deployment-platforms)
5. [Testing the Pipeline](#testing-the-pipeline)

---

## 🎯 Overview

The CI/CD pipeline automates:
- ✅ Code testing and linting
- 🐳 Docker image building
- 🚀 Automatic deployment to Render
- 📊 Health checks and notifications

### Pipeline Flow:
```
Code Push → Tests → Build Docker Image → Deploy to Render → Health Check → Notify
```

---

## ⚙️ Prerequisites

1. **GitHub Account** with repository access
2. **Docker Hub Account** (free tier works)
3. **Render Account** (free tier available)
4. Git installed locally

---

## 🔐 GitHub Secrets Setup

You need to add the following secrets to your GitHub repository:

### Step 1: Go to Repository Settings
1. Navigate to your GitHub repository
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**

### Step 2: Add Required Secrets

#### 1. Docker Hub Credentials
```
Name: DOCKER_USERNAME
Value: your-dockerhub-username

Name: DOCKER_PASSWORD
Value: your-dockerhub-password (or access token)
```

**How to get Docker Hub Access Token:**
1. Login to [Docker Hub](https://hub.docker.com/)
2. Go to Account Settings → Security
3. Click "New Access Token"
4. Name it "GitHub Actions" and copy the token

#### 2. Render Deployment Hook
```
Name: RENDER_DEPLOY_HOOK_URL
Value: https://api.render.com/deploy/srv-xxxxx?key=xxxxx
```

**How to get Render Deploy Hook:**
1. Login to [Render](https://render.com/)
2. Go to your service → Settings
3. Scroll to "Deploy Hook"
4. Click "Create Deploy Hook"
5. Copy the URL

#### 3. Render App URL
```
Name: RENDER_APP_URL
Value: https://your-app-name.onrender.com
```

This is the URL where your app will be deployed.

---

## 🌐 Deployment Platforms

### Option 1: Deploy to Render (Recommended)

**Step 1: Create Render Account**
1. Go to [Render.com](https://render.com/)
2. Sign up with GitHub

**Step 2: Create Web Service**
1. Click "New" → "Web Service"
2. Connect your GitHub repository
3. Configure:
   ```
   Name: customer-churn-prediction
   Environment: Docker
   Branch: master
   Plan: Free
   ```

**Step 3: Environment Variables**
Add these in Render dashboard:
```
PYTHON_VERSION=3.9
PORT=8501
```

**Step 4: Deploy**
1. Click "Create Web Service"
2. Render will automatically build and deploy
3. Get your deploy hook URL from Settings

---

### Option 2: Deploy to Heroku

**Create `Procfile`:**
```
web: streamlit run src/app/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

**Deploy Commands:**
```bash
heroku login
heroku create customer-churn-app
git push heroku master
```

---

### Option 3: Deploy to AWS (EC2)

**SSH into EC2 instance:**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

# Install Docker
sudo apt-get update
sudo apt-get install docker.io -y

# Pull and run container
sudo docker pull yourusername/customer-churn:latest
sudo docker run -d -p 8501:8501 yourusername/customer-churn:latest
```

---

## 🧪 Testing the Pipeline

### 1. Test Locally First

**Build Docker Image:**
```bash
cd Customer-churn
docker build -t customer-churn:test .
```

**Run Container:**
```bash
docker run -p 8501:8501 customer-churn:test
```

**Test Application:**
Open browser: http://localhost:8501

### 2. Test CI/CD Pipeline

**Trigger Pipeline:**
```bash
# Make a small change
echo "# Test deployment" >> README.md
git add .
git commit -m "test: trigger CI/CD pipeline"
git push origin master
```

**Monitor Pipeline:**
1. Go to GitHub repository
2. Click "Actions" tab
3. Watch the workflow run
4. Check each job status

### 3. Verify Deployment

**Check Render Dashboard:**
1. Login to Render
2. Go to your service
3. Check "Logs" tab
4. Wait for "Application ready" message

**Test Live Application:**
```bash
# Check health endpoint
curl https://your-app.onrender.com/_stcore/health

# Open in browser
https://your-app.onrender.com
```

---

## 📊 Pipeline Jobs Explained

### Job 1: Test and Lint
- ✅ Runs `pytest` for unit tests
- ✅ Runs `flake8` for code linting
- ✅ Runs `black` for code formatting check
- ✅ Generates coverage report

### Job 2: Build Docker Image
- 🐳 Builds Docker image from Dockerfile
- 🐳 Pushes to Docker Hub with tags:
  - `latest` (most recent)
  - `<commit-sha>` (version specific)

### Job 3: Deploy to Render
- 🚀 Triggers Render deployment hook
- ⏳ Waits for deployment to complete
- 🏥 Performs health check
- ✅ Confirms application is running

### Job 4: Notify Status
- 📧 Sends deployment status notification
- ✅ Success or ❌ Failure

---

## 🔍 Troubleshooting

### Issue: Tests Failing
```bash
# Run tests locally
pytest tests/ -v

# Check specific test
pytest tests/test_train.py -v
```

### Issue: Docker Build Failing
```bash
# Check Dockerfile syntax
docker build -t test .

# View build logs
docker build --no-cache -t test . 2>&1 | tee build.log
```

### Issue: Deployment Failing
```bash
# Check Render logs
# Go to Render Dashboard → Your Service → Logs

# Check health endpoint
curl https://your-app.onrender.com/_stcore/health
```

### Issue: App Not Loading
1. Check if port 8501 is exposed
2. Check Streamlit configuration
3. Check environment variables
4. Review application logs

---

## 🎯 Best Practices

### 1. Version Control
```bash
# Always use meaningful commit messages
git commit -m "feat: add new prediction feature"
git commit -m "fix: resolve model loading issue"
git commit -m "docs: update deployment guide"
```

### 2. Testing
```bash
# Run tests before pushing
pytest tests/ -v

# Check code quality
flake8 src/
black src/ --check
```

### 3. Monitoring
- Check GitHub Actions after each push
- Monitor Render logs for errors
- Set up error alerting

### 4. Security
- Never commit secrets to Git
- Use environment variables
- Rotate access tokens regularly

---

## 📈 Next Steps

1. ✅ Set up GitHub secrets
2. ✅ Test pipeline with a commit
3. ✅ Monitor deployment
4. ✅ Share your app URL!

---

## 🆘 Support

If you encounter issues:
1. Check GitHub Actions logs
2. Check Render deployment logs
3. Review this guide
4. Check Docker Hub for image status

---

## 📝 Summary Checklist

- [ ] GitHub repository created and code pushed
- [ ] Docker Hub account created
- [ ] Render account created
- [ ] GitHub secrets configured (4 secrets)
- [ ] Render web service created
- [ ] Deploy hook obtained
- [ ] CI/CD pipeline tested
- [ ] Application accessible online

**Your app will be live at:** `https://your-app-name.onrender.com`

---

## 🎉 Congratulations!

You now have a fully automated CI/CD pipeline for your Customer Churn Prediction application!

Every push to `master` will:
1. Run tests
2. Build Docker image
3. Deploy to production
4. Verify health
5. Notify you of status
