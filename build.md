```

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
venv一式はvercelには不要


git init
git branch -M main
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/ユーザー名/リポジトリ名.git
git push -u origin main


used model
facebook/blenderbot-90M


vercel deploy失敗
Running build in Washington, D.C., USA (East) – iad1
Cloning github.com/tztechno/vercel_flask_chatbot (Branch: main, Commit: 0e26ff3)
Skipping build cache, deployment was triggered without cache.
Cloning completed: 210.000ms
Running "vercel build"
Vercel CLI 40.1.0
WARN! Due to `builds` existing in your configuration file, the Build and Development Settings defined in your Project Settings will not apply. Learn More: https://vercel.link/unused-build-settings
Installing required dependencies...
Build Completed in /vercel/output [4m]
Deploying outputs...
Failed to process build result for "app.py". Data: {"type":"Lambda"}.
Error: data is too long

```
