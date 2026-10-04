@echo off
rem Hugging Face login for bandscribe. The token is stored under <project>\data\hf (not in AppData).
for %%I in ("%~dp0..") do set "HF_HOME=%%~fI\data\hf"
call "%~dp0uvw.cmd" tool run --from huggingface_hub hf auth login %*
echo.
call "%~dp0uvw.cmd" tool run --from huggingface_hub hf auth whoami
