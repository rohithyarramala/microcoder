#!/usr/bin/env bash
# One-shot ESP32 lab runner. Run from the repository root:
#
#   bash esp32/run_lab.sh sim            # host simulation only (needs gcc)
#   bash esp32/run_lab.sh qemu           # ESP-IDF build + Espressif QEMU (functional check)
#   bash esp32/run_lab.sh board /dev/ttyACM0   # real ESP32-S3: build, flash, capture log
#
# Every mode ends by scoring the log with the paper's unit tests.
set -euo pipefail
MODE=${1:-sim}
CKPT=${CKPT:-ckpt/s_s0.pt}
SPLIT=${SPLIT:-test_id,test_ood}
IDF_VER=${IDF_VER:-v5.4.2}

python3 scripts/make_firmware_assets.py "$CKPT" "$SPLIT"

need_idf() {
  if [ -z "${IDF_PATH:-}" ]; then
    if [ ! -d "$HOME/esp-idf" ]; then
      git clone -b "$IDF_VER" --depth 1 --recursive --shallow-submodules https://github.com/espressif/esp-idf.git "$HOME/esp-idf"
      "$HOME/esp-idf/install.sh" esp32s3
    fi
    # shellcheck disable=SC1091
    . "$HOME/esp-idf/export.sh"
  fi
}

case "$MODE" in
  sim)
    make -C esp32/host_sim
    ./esp32/host_sim/sim > lab_sim_log.txt
    python3 scripts/score_device_log.py lab_sim_log.txt "$SPLIT"
    ;;
  qemu)
    need_idf
    python3 "$IDF_PATH/tools/idf_tools.py" install qemu-xtensa
    . "$IDF_PATH/export.sh"
    ( cd esp32/firmware && idf.py set-target esp32s3 && idf.py build )
    # QEMU timing is NOT cycle-accurate: use this mode to check functionality only.
    ( cd esp32/firmware && timeout 3600 idf.py qemu monitor ) | tee lab_qemu_log.txt || true
    python3 scripts/score_device_log.py lab_qemu_log.txt "$SPLIT"
    ;;
  board)
    PORT=${2:?serial port, e.g. /dev/ttyACM0 or COM5}
    need_idf
    ( cd esp32/firmware && idf.py set-target esp32s3 && idf.py build && idf.py -p "$PORT" flash )
    python3 - "$PORT" <<'EOF'
import sys, serial, time
s = serial.Serial(sys.argv[1], 115200, timeout=1)
s.setDTR(False); s.setRTS(True); time.sleep(0.1); s.setRTS(False)   # reset into the app
with open("lab_board_log.txt", "w") as f:
    t0 = time.time()
    while time.time() - t0 < 7200:
        line = s.readline().decode(errors="replace")
        if line:
            f.write(line); f.flush(); print(line, end="")
        if line.startswith("MC_DONE"):
            break
EOF
    python3 scripts/score_device_log.py lab_board_log.txt "$SPLIT"
    ;;
  *) echo "usage: $0 sim|qemu|board <port>"; exit 1 ;;
esac
