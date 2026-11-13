# ESP32 lab

Everything here runs the **same C engine** (`engine/microcoder.c`) and the
**same INT8 model blob** that the paper evaluates on the host.

| Folder | What it is | Where it runs |
|---|---|---|
| `engine/` | INT8 inference engine + byte-level BPE tokenizer, portable C99 | everywhere |
| `host_sim/` | Runs the unmodified firmware `app_main()` on a PC with ESP32-S3 memory budgets (300 KiB free internal heap, configurable PSRAM) | Linux / macOS |
| `firmware/` | ESP-IDF benchmark firmware: tokenises, generates and times every benchmark prompt, prints code over UART | ESP32-S3 board or Wokwi (not Espressif QEMU, see below) |
| `mp_module/` | MicroPython native module `microcoder` (for the on-device REF loop in `../app/`) | ESP32-S3 running MicroPython |

## 1. Embed the model and prompts

```bash
python scripts/make_firmware_assets.py ckpt/m1_s0.pt test_ood     # or test_id
```

## 2. Host simulation (no hardware needed)

```bash
make -C esp32/host_sim && ./esp32/host_sim/sim > sim_log.txt        # 8 MiB PSRAM
./esp32/host_sim/sim 0                                              # board without PSRAM: allocation fails
python scripts/score_device_log.py sim_log.txt
```

## 3. Real ESP32-S3 (recommended: ESP32-S3-DevKitC-1 N8R8, 8 MB flash + 8 MB PSRAM)

```bash
. $IDF_PATH/export.sh            # ESP-IDF v5.x
cd esp32/firmware
idf.py set-target esp32s3 build flash monitor | tee ../../device_log.txt
cd ../.. && python scripts/score_device_log.py device_log.txt
```

`score_device_log.py` re-scores the code printed by the chip with the paper's
unit tests and reports median decode ms/token, prefill ms/token and end-to-end
latency. **These are the numbers that go into Table V of the paper.**

## 4. Simulators

* **Wokwi** (VS Code extension or `wokwi-cli`): build with `idf.py build`, then
  start the simulator in `esp32/firmware` (uses `wokwi.toml` + `diagram.json`,
  ESP32-S3 with 8 MB PSRAM). Wokwi is functionally exact (same printed code,
  same accuracy) but its timing is *not* cycle-accurate — do not report Wokwi
  latency as hardware latency.
* **Espressif QEMU: not supported.** The QEMU bundled with ESP-IDF v5.4.2 (`esp-develop-9.0.0-20240606`) fails the ESP32-S3 PSRAM ID read in both octal and quad mode, so ESP-IDF aborts before `app_main()`; the model state needs PSRAM. CI (`.github/workflows/esp32-build.yml`) therefore builds the firmware, records `idf.py size`, and checks function with the host simulation instead.
  Same caveat: functional check only.

## 5. MicroPython with the on-device REF loop

```bash
cd micropython/ports/esp32
make BOARD=ESP32_GENERIC_S3 BOARD_VARIANT=SPIRAM_OCT USER_C_MODULES=/path/to/esp32/mp_module/micropython.cmake
# first: TARGET=mp python scripts/make_firmware_assets.py ckpt/deploy.pt   (application model)
# flash, then: mpremote cp app/ref.py app/lint.py app/assistant.py :
```

Here the verification gate is MicroPython's own `compile()`.
