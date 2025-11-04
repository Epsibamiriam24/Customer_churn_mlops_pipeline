# Automatic Deployment Guide

## Current Deployment Architecture

Your application uses a **two-layer deployment system**:

### Layer 1: Render Direct Integration (Automatic)
- Render watches your GitHub repository
- Every push to `master` branch automatically triggers a build
- No additional configuration needed (just connect your repo)

### Layer 2: GitHub Actions + Render Webhook (Optional)
- GitHub Actions tests and validates your code
- On success, sends webhook to Render to trigger deployment
- Requires: `RENDER_DEPLOY_HOOK_URL` secret in GitHub

---

## Setup Steps

### Step 1: Ensure Render is Connected to Your GitHub Repo

1. Go to https://dashboard.render.com
2. Click on your `customer-churn-prediction` service
3. Check **Settings** → **Source**
4. Should show: `https://github.com/Epsibamiriam24/Customer_churn_mlops_pipeline.git`
5. If not connected, click "Connect Repository"

### Step 2: Enable Auto-Deployment (Optional but Recommended)

To speed up deployment and get GitHub Actions confirmation:

1. In Render service, go to **Settings**
2. Scroll to "Deploy Hook"
3. Copy the URL (looks like: `https://api.render.com/deploy/srv-xxxxx?key=xxxxx`)
4. Go to GitHub repo → **Settings** → **Secrets and variables** → **Actions**
5. Create new secret:
   - Name: `RENDER_DEPLOY_HOOK_URL`
   - Value: (paste the Render deploy hook URL)

### Step 3: How Automatic Deployment Works

```
You push code to GitHub master branch
           ↓
GitHub Actions CI/CD runs (test, build)
           ↓
If RENDER_DEPLOY_HOOK_URL is set:
   → GitHub sends webhook to Render
   → Render receives deployment trigger
           ↓
Render builds Docker image
           ↓
Render deploys application
           ↓
Your app is LIVE at: https://customer-churn-prediction-xxxx.onrender.com
```

---

## Current Status

### Automatic Deployment Methods:

**Method 1: Direct Render Integration** ✅ ENABLED
- Any push to master → Render automatically builds
- No webhook needed
- Takes 5-10 minutes

**Method 2: GitHub Actions + Webhook** ⏳ OPTIONAL
- Requires RENDER_DEPLOY_HOOK_URL secret
- Gives faster feedback in GitHub
- Takes 2-3 minutes after GitHub Actions passes

---

## Troubleshooting

### App not deploying?

1. **Check Render Dashboard**
   - Go to https://dashboard.render.com
   - Look for deployment status
   - Check logs for build errors

2. **Check GitHub Actions**
   - Go to https://github.com/Epsibamiriam24/Customer_churn_mlops_pipeline/actions
   - Click latest workflow run
   - Check if all jobs passed

3. **Check if Render is connected**
   - Render service → Settings → Source
   - Should show your GitHub repo URL
   - If showing red X, reconnect the repository

4. **Check if buildFilter is removed**
   - `render.yaml` should NOT have buildFilter
   - If present, remove it and commit

### Deployment takes too long?

- First build: 10-15 minutes (normal)
- Subsequent builds: 5-10 minutes (Docker cache)
- Render free tier: May be slower than paid plans

### Test failures blocking deployment?

- Current CI/CD continues even if tests fail
- Build still happens: `continue-on-error: true`
- If you want strict testing, let us know

---

## Manual Deploy (If Needed)

If automatic deployment fails:

1. **Via Render Dashboard**
   - Go to Render service
   - Click "Manual Deploy"
   - Select "Deploy latest commit"

2. **Via GitHub Actions**
   - Click "Run workflow"
   - Select branch: `master`
   - Click "Run workflow"

---

## Testing Your Deployment

Once deployed:

1. **Access your app**
   ```
   https://customer-churn-prediction-xxxx.onrender.com
   ```
   (Check Render dashboard for exact URL)

2. **Test the features**
   - Enter customer name
   - Fill prediction form
   - Click "Predict Churn"
   - Download PDF report

3. **Check logs**
   - Render Dashboard → Logs
   - Should show Streamlit starting on port 8501

---

## Next Steps

1. ✅ Push code to GitHub master
2. ✅ Watch GitHub Actions run tests
3. ✅ Watch Render build and deploy
4. ✅ Your app is live!

---

## Quick Reference

| Component | Status | Link |
|-----------|--------|------|
| GitHub Repo | ✅ Connected | https://github.com/Epsibamiriam24/Customer_churn_mlops_pipeline |
| GitHub Actions | ✅ Configured | https://github.com/Epsibamiriam24/Customer_churn_mlops_pipeline/actions |
| Render Service | ✅ Connected | https://dashboard.render.com |
| Auto Deploy Hook | ⏳ Optional | GitHub Secrets → RENDER_DEPLOY_HOOK_URL |
| Render.yaml | ✅ Configured | buildFilter removed for all commits |

---

## Need Help?

Common issues and solutions:
- Render build fails → Check Render logs
- GitHub Actions fail → Check test logs
- App not responding → Check Render service is running
- Port issues → Confirm PORT=8501 in render.yaml

