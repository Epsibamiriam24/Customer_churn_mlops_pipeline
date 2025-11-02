# 🚀 Quick Deployment Setup Guide

## Automatic Deployment is Already Configured! ✅

Your CI/CD pipeline is ready to deploy automatically to Render. Follow these steps:

---

## Step 1: Create Render Account (2 minutes)

1. Go to https://render.com/
2. Click "Get Started for Free"
3. Sign up with your GitHub account (recommended)

---

## Step 2: Create Web Service (3 minutes)

1. Click "New +" → "Web Service"
2. Connect your GitHub repository: `Customer_churn_mlops_pipeline`
3. Configure the service:
   ```
   Name: customer-churn-prediction
   Environment: Docker
   Branch: master
   Plan: Free (or Starter)
   ```
4. Click "Create Web Service"

---

## Step 3: Get Deploy Hook (1 minute)

1. In your Render service, go to "Settings"
2. Scroll down to "Deploy Hook"
3. Click "Create Deploy Hook"
4. **Copy the URL** (looks like: `https://api.render.com/deploy/srv-xxxxx?key=xxxxx`)

---

## Step 4: Add GitHub Secret (1 minute)

1. Go to your GitHub repo: https://github.com/Epsibamiriam24/Customer_churn_mlops_pipeline
2. Click "Settings" → "Secrets and variables" → "Actions"
3. Click "New repository secret"
4. Add secret:
   ```
   Name: RENDER_DEPLOY_HOOK_URL
   Value: [paste the deploy hook URL you copied]
   ```
5. Click "Add secret"

---

## Step 5: Test Deployment (30 seconds)

1. Make any small change to trigger deployment:
   ```bash
   echo "# Deployment test" >> README.md
   git add README.md
   git commit -m "test: trigger deployment"
   git push origin master
   ```

2. Watch the magic happen! ✨
   - GitHub Actions: https://github.com/Epsibamiriam24/Customer_churn_mlops_pipeline/actions
   - Render Dashboard: https://dashboard.render.com/

---

## 🎉 Your App Will Be Live At:

```
https://customer-churn-prediction.onrender.com
```

(The exact URL will be shown in your Render dashboard)

---

## 📊 What Happens on Every Push:

```
Code Push → GitHub Actions
           ↓
        Run Tests ✅
           ↓
    Build Docker Image ✅
           ↓
   Trigger Render Deploy 🚀
           ↓
    Render builds & deploys
           ↓
     App is LIVE! 🎉
```

---

## ⚡ Quick Commands:

### Deploy manually (without code changes):
```bash
curl -X POST YOUR_DEPLOY_HOOK_URL
```

### Check deployment status:
```bash
# Visit Render dashboard
# Or check GitHub Actions tab
```

---

## 🔍 Troubleshooting:

**Q: Deployment not triggering?**
- Check if RENDER_DEPLOY_HOOK_URL secret is set correctly
- Make sure you're pushing to `master` branch

**Q: Build failing on Render?**
- Check Render logs in the dashboard
- Ensure Dockerfile is working (it should be!)

**Q: App not loading?**
- Render free tier takes 30-60 seconds for first load
- Check if deployment completed in Render dashboard

---

## 🎯 You're All Set!

Your application now has:
- ✅ Automated testing
- ✅ Docker containerization
- ✅ Continuous deployment
- ✅ Free cloud hosting

Every time you push code, it will automatically:
1. Run tests
2. Build Docker image
3. Deploy to production

**Total setup time: ~7 minutes** 🚀

---

## 📝 Optional: Add More Secrets (Later)

For Docker Hub push (optional):
```
DOCKER_USERNAME: your-dockerhub-username
DOCKER_PASSWORD: your-dockerhub-password
```

For health checks (optional):
```
RENDER_APP_URL: https://your-app.onrender.com
```
