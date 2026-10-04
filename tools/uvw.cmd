@echo off
rem uv wrapper: keeps python installs and caches inside the project folder (outside the app sandbox).
rem Paths are relative to this file (tools\..), so the wrapper keeps working if the folder moves.
for %%I in ("%~dp0..") do set "_UVW_ROOT=%%~fI"
set "UV_CACHE_DIR=%_UVW_ROOT%\data\uv-cache"
set "UV_PYTHON_INSTALL_DIR=%_UVW_ROOT%\tools\python"
set "_UVW_ROOT="
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_LINK_MODE=copy"
"%~dp0uv.exe" %*
