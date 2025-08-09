#!/bin/bash

LOG_FILE="docs/setup_log.md"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

if [ -z "$1" ]; then
  echo "Usage: ./log_setup.sh "Your custom log message here""
  exit 1
fi

echo "## 📝 Auto Log Entry — $DATE" >> $LOG_FILE
echo "" >> $LOG_FILE
echo "- [ ] $1" >> $LOG_FILE
echo "" >> $LOG_FILE

echo "Logged: $1"
