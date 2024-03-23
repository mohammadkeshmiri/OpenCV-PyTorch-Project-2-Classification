$path="C:\Program Files\Docker\Docker\frontend\Docker Desktop.exe"
if (-not(get-process | ?{$_.path -eq $path}))
{
    Write-Output $P "Starting Docker Desktop..."
    Start-Process $path -WindowStyle Hidden -Wait
}