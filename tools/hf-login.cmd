@echo off
rem Hugging Face login for gtab. The token is stored under D:\gtab\data\hf (not in AppData).
set "HF_HOME=D:\gtab\data\hf"
call "%~dp0uvw.cmd" tool run --from huggingface_hub hf auth login %*
echo.
call "%~dp0uvw.cmd" tool run --from huggingface_hub hf auth whoami
