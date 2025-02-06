```
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt

venv一式はvercelには不要




git rev-parse --abbrev-ref HEAD

ls -a

git init

git checkout -b main

git add .
git commit -m "Initial commit"

git remote add origin https://github.com/tztechno/vercel_flask_chatbot.git

git remote -v

git push -u origin main



git stash

git pull --rebase origin main

git stash pop

git add .
git commit -m "Resolve conflicts and update"

git push origin main

```