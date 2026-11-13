#!/usr/bin/env bash
# Run the built ESP32-S3 firmware in Espressif QEMU without a TTY (CI-safe).
# Mirrors what `idf.py qemu` does in ESP-IDF v5.4.2 (tools/idf_py_actions/qemu_ext.py): merge the flash
# image, write the default eFuse image, launch qemu-system-xtensa -M esp32s3. Adds -m 4M for PSRAM (QEMU's
# maximum; the board has 8 MB). Do NOT pass strap_mode=0x07: that is idf.py's --boot-mode (UART download mode). Serial goes to a file; we stop on MC_DONE, a panic/abort, or a timeout.
# QEMU is not cycle-accurate: its timing must never be reported as latency.
#
#   . $IDF_PATH/export.sh && bash esp32/qemu_run.sh lab_qemu_log.txt [timeout_s]
set -uo pipefail
LOG=${1:-lab_qemu_log.txt}
TMO=${2:-5400}
PSRAM=${QEMU_PSRAM:-4M}   # Espressif QEMU: "RAM size larger than 4 MB not supported"
LOG=$(realpath -m "$LOG")
B=esp32/firmware/build

( cd "$B" && python -m esptool --chip=esp32s3 merge_bin --output=qemu_flash.bin --fill-flash-size=8MB @flash_args ) || exit 1
# default eFuse image used by idf.py qemu for esp32s3 (chip revision 0.3): 1 byte 0x0c at offset 38, rest zero
python - "$B/qemu_efuse.bin" <<'EOF'
import sys
e = bytearray(1024); e[38] = 0x0c
open(sys.argv[1], "wb").write(bytes(e))
EOF

ARGS=(-M esp32s3 -m "$PSRAM"
      -drive "file=$B/qemu_flash.bin,if=mtd,format=raw"
      -drive "file=$B/qemu_efuse.bin,if=none,format=raw,id=efuse"
      -global driver=nvram.esp32c3.efuse,property=drive,value=efuse
      -global driver=timer.esp32s3.timg,property=wdt_disable,value=true
      -nic user,model=open_eth -nographic -no-reboot
      -serial "file:$LOG" -monitor none)
echo "qemu-system-xtensa ${ARGS[*]}"
: > "$LOG"
qemu-system-xtensa "${ARGS[@]}" &
QPID=$!
START=$(date +%s); RC=2
while kill -0 "$QPID" 2>/dev/null; do
  if grep -q '^MC_DONE' "$LOG"; then RC=0; break; fi
  if grep -qE 'Guru Meditation|abort\(\) was called|PSRAM.*(fail|not found)|MC_ERROR|wait uart0 download' "$LOG"; then RC=1; sleep 2; break; fi
  # fail fast if the app never starts (no MC_INFO after 15 min) instead of burning the whole timeout
  if [ $(( $(date +%s) - START )) -ge 900 ] && ! grep -q '^MC_INFO' "$LOG"; then echo "no MC_INFO after 900s"; RC=4; break; fi
  if [ $(( $(date +%s) - START )) -ge "$TMO" ]; then echo "timeout after ${TMO}s"; RC=3; break; fi
  sleep 10
  NOW=$(( $(date +%s) - START )); [ $((NOW % 120)) -lt 10 ] && echo "[${NOW}s] $(grep -c '^MC_END' "$LOG") prompts done"
done
kill "$QPID" 2>/dev/null; wait "$QPID" 2>/dev/null
echo "exit=$RC; $(grep -c '^MC_END' "$LOG") prompts in log"
head -c 3000 "$LOG"; echo; echo ...; tail -n 15 "$LOG"
exit $RC
