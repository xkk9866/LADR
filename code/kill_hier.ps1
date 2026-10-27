Get-CimInstance Win32_Process -Filter "name='bash.exe'" | Where-Object { $_.CommandLine -match 'run_hier\.sh' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; 'killed' }
