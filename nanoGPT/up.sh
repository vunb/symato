mv _git .git
git commit -am "change"
git config pull.rebase false
git pull
mv .git _git