# git

## set-up
```
type -p curl >/dev/null || (sudo apt update && sudo apt install curl -y)
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```
From: https://github.com/cli/cli/blob/trunk/docs/install_linux.md

## login
```
gh auth login
```

## configure user.name
```
git config --global user.name "username"
git config --get user.name
```

## configure user.email
```
git config --global user.email "youremail@yourprovider.com" 
git config --get user.email
```