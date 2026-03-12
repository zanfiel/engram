# Run this as Administrator
New-NetFirewallRule -DisplayName "Engram Memory Server" -Direction Inbound -LocalPort 4200 -Protocol TCP -Action Allow -Profile Private,Domain -Description "Engram memory server - LAN and Headscale access"
Write-Host "Firewall rule added." -ForegroundColor Green
