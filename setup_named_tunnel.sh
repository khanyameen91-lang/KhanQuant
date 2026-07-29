#!/bin/bash
# setup_named_tunnel.sh
# Replaces the temporary quick tunnel with the permanent named "tastybot" tunnel

set -e

echo "=== Step 1: Stop & remove old quick-tunnel cloudflared service ==="
sudo systemctl stop cloudflared 2>/dev/null && echo "Stopped" || echo "Was not running"
sudo systemctl disable cloudflared 2>/dev/null && echo "Disabled" || echo "Was not enabled"

# Remove old service file if it was manually created
if [ -f /etc/systemd/system/cloudflared.service ]; then
    sudo rm /etc/systemd/system/cloudflared.service
    sudo systemctl daemon-reload
    echo "Old service file removed"
fi

echo ""
echo "=== Step 2: Install named tunnel as a system service ==="
sudo cloudflared service install eyJhIjoiMTRjMWQ2ZDU2ZDhjYWU4M2E2OTA3Nzg2Zjk2ZWNmMTQiLCJ0IjoiMDMwYTViNGEtNTEwNS00OTI3LWE3MjQtMDNmM2Y5ZjVhNWZhIiwicyI6Ik0yUTVZV1kwTVdJdFl6Z3haaTAwTjJKbUxUbG1OalF0TkRKaE9UZ3haREEyTVRVeCJ9

echo ""
echo "=== Step 3: Start and enable the service ==="
sudo systemctl start cloudflared
sudo systemctl enable cloudflared
sleep 6

echo ""
echo "=== Step 4: Status check ==="
sudo systemctl status cloudflared --no-pager | head -20

echo ""
echo "=== Step 5: Tunnel connection logs ==="
sudo journalctl -u cloudflared -n 25 --no-pager

echo ""
echo "=== Done ==="
echo "If you see 'Registered tunnel connection' above, the tunnel is live."
echo "Next: set the public hostname in the Cloudflare dashboard."
