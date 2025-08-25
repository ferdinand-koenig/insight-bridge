#!/bin/bash
# swap size in GB
SWAP_SIZE_GB=6
SWAP_FILE=/swapfile

# Check if swap file already exists
if [ ! -f $SWAP_FILE ]; then
    echo "Creating swap file of ${SWAP_SIZE_GB}G..."
    fallocate -l ${SWAP_SIZE_GB}G $SWAP_FILE
    chmod 600 $SWAP_FILE
    mkswap $SWAP_FILE
fi

# Enable swap
swapon $SWAP_FILE

# Set swappiness to 2
sysctl vm.swappiness=2

# Persist swappiness for next reboots
if ! grep -q "vm.swappiness" /etc/sysctl.conf; then
    echo "vm.swappiness=2" >> /etc/sysctl.conf
fi

# Persist swap file in /etc/fstab if not already present
if ! grep -q "$SWAP_FILE" /etc/fstab; then
    echo "$SWAP_FILE none swap sw 0 0" >> /etc/fstab
fi
