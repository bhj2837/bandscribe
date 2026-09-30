@echo off
rem uv wrapper: keeps python installs and caches on D: (outside the app sandbox)
set "UV_CACHE_DIR=D:\gtab\data\uv-cache"
set "UV_PYTHON_INSTALL_DIR=D:\gtab\tools\python"
set "UV_PYTHON_PREFERENCE=only-managed"
set "UV_LINK_MODE=copy"
"%~dp0uv.exe" %*
