call .\build.bat

@echo off
:choice
set /P c=I will now recursively delete the contents of  %~dp0..\inference\artifact\  and   %2 . Are you sure you want to continue? (type yes)? 
if /I "%c%" EQU "yes" goto deleteandrun
echo aborting
exit

:deleteandrun

rmdir /S "%~dp0..\inference\artifact\"
mkdir "%~dp0..\inference\artifact\"
rmdir /S %2
mkdir %2

docker run --rm --gpus all^
 --memory $MEMORY --memory-swap $MEMORY^
 --cap-drop ALL --cap-add SYS_NICE --security-opt "no-new-privileges"^
 --network none --shm-size 32g --pids-limit 1024^
 -v %1:\input\:ro^
 -v %~dp0..\inference\artifact\:\output\^
 -v %2:\scratch\^
 stoictrain
