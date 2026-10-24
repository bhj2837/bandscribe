@echo off
rem Double-click: the song list of the score viewer opens in the browser. Close this window to stop it.
title bandscribe viewer
"%~dp0.venv\Scripts\python.exe" -m bandscribe.cli view %*
if errorlevel 1 pause
