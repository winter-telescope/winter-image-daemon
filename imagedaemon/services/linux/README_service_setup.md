# WINTER Image Daemon — System Service Setup (Linux)

This sets up the Image Daemon as a **system-wide** `systemd` service under `/etc/systemd/system`, similar to other services you run, and executes as user `winter`.

Repo paths assumed:

- Service unit: `imagedaemon/services/linux/imagedaemon.service`
- Launcher script: `imagedaemon/services/linux/run_imagedaemon.sh`
- Repo root: `/home/winter/GIT/winter-image-daemon`
- Conda env: `/home/winter/GIT/winter-image-daemon/.conda`

## 1) Make launcher executable

```bash
chmod +x /home/winter/GIT/winter-image-daemon/imagedaemon/services/linux/run_imagedaemon.sh

```
## 2) Install the systemd unit
Copy the service file into your user systemd directory:

```bash:
sudo cp /home/winter/GIT/winter-image-daemon/services/linux/imagedaemon.service /etc/systemd/system/imagedaemon.service
sudo systemctl daemon-reload
sudo systemctl enable --now imagedaemon.service

```

## 3) Check the status/logs
Check status:

```bash:
sudo systemctl status imagedaemon.service
sudo journalctl -u imagedaemon.service -f
```


## 4) Customizing arguments (cameras, NameServer host, logfile)
Edit `/etc/systemd/system/imagedaemon.service` and change the `ExecStart` line's arguments:

```ini:
ExecStart=/home/winter/GIT/winter-image-daemon/services/linux/run_imagedaemon.sh \
  --cameras winter,qcmos,summer-ccd,pirt \
  -n 192.168.1.10 \
  --logfile /home/winter/data/imagedaemon.log

```

Then apply changes:
```bash:
sudo systemctl daemon-reload
sudo systemctl restart imagedaemon.service
```

**Alternative (preferred):** Use an override without touching the original file:
```bash:
sudo systemctl edit imagedaemon.service
```

and paste:
```ini:
[Service]
ExecStart=
ExecStart=/home/winter/GIT/winter-image-daemon/services/linux/run_imagedaemon.sh \
  --cameras winter,qcmos,summer-ccd,pirt \
  -n 192.168.1.10 \
  --logfile /home/winter/data/imagedaemon.log
```

then:
```bash:
sudo systemctl daemon-reload
sudo systemctl restart imagedaemon.service
```

## 5) Updating the daemon:
When you pull in new code:

```bash:
cd /home/winter/GIT/winter-image-daemon
git pull
sudo systemctl restart imagedaemon.service
```

If you changed the unit file:
```bash:
cp servicec/imagedaemon.service ~/.config/systemd/user/imagedaemon.service
sudo systemctl daemon-reload
sudo systemctl restart imagedaemon.service
```

## 7) Uninstall

```bash:
sudo systemctl disable --now imagedaemon.service
rm -f ~/.config/systemd/user/imagedaemon.service
sudo systemctl daemon-reload
```

## 8) Notes:
- The service runs as user winter (set in the unit). Ensure the WorkingDirectory and paths are accessible by that user.
- The launcher script activates Conda before running the console entrypoint so you don’t need to inject environment into the unit.
- If your Conda path is nonstandard, edit run_imagedaemon.sh to point to the correct conda.sh.