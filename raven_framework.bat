@echo off
ECHO ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ECHO[
ECHO       .---.        .------######       #####     ###   ###  ########  ###    ###
ECHO      /     \  __  /    --###  ###    ###  ###   ###   ###  ###       #####  ###
ECHO     / /     \(  )/    --###  ###    ###   ###  ###   ###  ######    ### ######
ECHO    //////   ' \/ `   --#######     #########  ###   ###  ###       ###  #####
ECHO   //// / // :    :   -###   ###   ###   ###    ######   ####      ###   ####
ECHO  // /   /  /`    '---###    ###  ###   ###      ###    ########  ###    ###
ECHO //          //..\\
ECHO ===========UU====UU=============================================================
ECHO            '//||\\`
ECHO              ''``
ECHO[
ECHO[
set /p input_file="Enter input file: "
ECHO[
FOR %%i IN ("%input_file%") DO (
ECHO *** INPUT FILE LOCATION ***
ECHO     filedrive=%%~di
ECHO     filepath=%%~pi
ECHO     filename=%%~ni
ECHO     fileextension=%%~xi
)
ECHO[
ECHO *** RUNNING ***:
ECHO[
Pushd %filepath%
bash.exe raven_framework %input_file%
ECHO[
ECHO[
pause
