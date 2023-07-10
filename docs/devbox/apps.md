# useful apps
#### <div style="text-align: left"> <a href="devbox.md"><b>back to devbox</b></a> <br/></div>

Most probably, everything in this repository could be handled with *vi*. Below are a handful of productivity tools I hope could help along the way.

## webapp manager
OneNote does not have a Linux client. Webapp manager wraps OneNote web interface in an application so you could add it to the dock.

Webapp manager allows running websites as an application.

### download 

    http://packages.linuxmint.com/pool/main/w/webapp-manager/ 

### install 

```
cd ~/Downloads/ 
sudo dpkg -i webapp-manager_1.3.2_all.deb   
sudo apt --fix-broken install 
sudo dpkg -i webapp-manager_1.3.2_all.deb   
sudo apt-get -f install   
```
 
### configure

```
sudo -s 
mkdir /usr/share/icons/OneNote 
#download Microsoft_Office_OneNote_\(2019–present\).svg.png 
mv Microsoft_Office_OneNote_\(2019–present\).svg.png /usr/share/icons/OneNote/ 
``` 

## set terminal title
I found it useful to label different terminal tabs with their main function, e.g. LA-BLD, to indicate that is the window where the llama-api container image is built.

### add to .bashrc.

```
# function to set terminal title   
 

function set-title() { 
  if [[ -z "$ORIG" ]]; then 
    ORIG=$PS1 
  fi 
  TITLE="\[\e]2;$*\a\]" 
  PS1=${ORIG}${TITLE} 
} 

``` 

### how to use

set-title TEST 

 