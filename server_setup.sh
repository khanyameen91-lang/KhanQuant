#!/bin/bash
# server_setup.sh — Run on Oracle Cloud server to set up the bot

set -e
echo "========================================="
echo "  Tasty Bot Server Setup"
echo "========================================="

# Update system
sudo apt-get update -qq
sudo apt-get install -y python3 python3-pip python3-venv wget curl

# Install system dependencies for pandas-ta / scikit-learn
sudo apt-get install -y python3-dev build-essential libopenblas-dev

# Install Python dependencies
cd ~/tastybot
pip3 install -r requirements.txt --break-system-packages 2>/dev/null || \
  pip3 install -r requirements.txt

# Verify key packages
python3 -c "import yfinance, pandas_ta, sklearn; print('All ML packages OK')" || \
  echo "Warning: some ML packages may not have installed correctly"

# Download cloudflared
if [ ! -f /usr/local/bin/cloudflared ]; then
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /tmp/cloudflared
  sudo mv /tmp/cloudflared /usr/local/bin/cloudflared
  sudo chmod +x /usr/local/bin/cloudflared
fi

# Create logs directory
mkdir -p ~/tastybot/logs

# Create systemd service for dashboard
sudo tee /etc/systemd/system/tasty-dashboard.service > /dev/null << 'EOF'
[Unit]
Description=Tasty Bot Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tastybot
ExecStart=/usr/bin/python3 dashboard.py
Restart=always
RestartSec=10
Environment=PYTHONIOENCODING=utf-8

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for bot
sudo tee /etc/systemd/system/tasty-bot.service > /dev/null << 'EOF'
[Unit]
Description=Tasty Bot Trading Bot
After=network.target tasty-dashboard.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tastybot
ExecStart=/usr/bin/python3 bot.py
Restart=always
RestartSec=30
Environment=PYTHONIOENCODING=utf-8

[Install]
WantedBy=multi-user.target
EOF

# Create systemd service for cloudflare tunnel
sudo tee /etc/systemd/system/tasty-tunnel.service > /dev/null << 'EOF'
[Unit]
Description=Tasty Bot Cloudflare Tunnel
After=network.target tasty-dashboard.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tastybot
ExecStart=/usr/local/bin/cloudflared tunnel --url http://localhost:5000 --logfile /home/ubuntu/tastybot/logs/tunnel.log
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start all services
sudo systemctl daemon-reload
sudo systemctl enable tasty-dashboard tasty-bot tasty-tunnel
sudo systemctl start tasty-dashboard
sleep 3
sudo systemctl start tasty-bot
sleep 2
sudo systemctl start tasty-tunnel

echo ""
echo "========================================="
echo "  Setup complete! Services running."
echo "========================================="
echo ""
echo "Waiting 15 seconds for Cloudflare URL..."
sleep 15
echo ""
echo "=== YOUR PUBLIC URL ==="
grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' ~/tastybot/logs/tunnel.log | tail -1
echo "======================="
echo ""
echo "Bookmark that URL on your phone!"
