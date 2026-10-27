$me = $PID
Get-CimInstance Win32_Process -Filter "name='bash.exe'" | Where-Object { $_.CommandLine -match 'code/run_all\.sh' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "killed bash $($_.ProcessId)" }
Get-CimInstance Win32_Process -Filter "name='python.exe'" | Where-Object { $_.CommandLine -match 'train\.py --name (gru|hier|abl_)' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "killed py $($_.ProcessId) $($_.CommandLine)" }
