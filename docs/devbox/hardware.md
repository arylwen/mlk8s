# Hardware queries

#### <div style="text-align: left"> <a href="../os.md"><b>back to operating system</b></a> <br/></div>

TLDR; - ROCm is not supported on Topaz XT.

## Graphics capabilities

    sudo apt install mesa-utils 
    glxinfo | grep -i vendor 

## Graphics hardware capabilities

    sudo lshw -C display 
    lspci -nnk | grep VGA -A 12 
    lspci -k | grep -iEA5 'vga|3d' 
    lspci -v 
    lspci -k | grep -iEA5 'vga|3d|display' 

## List OS and kernel info
```
uname -srmv
``` 
    Linux 5.15.0-76-generic #83~20.04.1-Ubuntu SMP Wed Jun 21 20:23:31 UTC 2023 x86_64