# development tools

## git
### config for commits
    git config --global user.name "FIRST_NAME LAST_NAME"
    git config --global user.email "MY_NAME@example.com"

Verify:

    git config --get user.name
    git config --get user.email 

## vscode
```
sudo snap install --classic code
```
On the dock, "show applicaitons" and search for VS Code. Once it starts, right click and 'add to favorites'.

### vs code extensions

1. python
2. jupyter

## dbeaver
Download community edition at:
https://dbeaver.io/download/

```
sudo snap install dbeaver-ce
```

Create a new connection and download the driver when prompted. On the postgresSQl tab check "show all databases".
