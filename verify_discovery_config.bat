@echo off
echo Activating virtual environment...
call conda activate cs2_agent

echo Verifying discovery configuration...
python train_discovery.py --config config/discovery_config.yaml --verify-config

echo Done.
pause 