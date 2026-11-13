# Host simulation of the ESP32 firmware: same main.c, same engine, same blobs.
FW = ../firmware/main
CFLAGS = -O2 -I. -I../engine -I$(FW)
sim: sim.c $(FW)/main.c ../engine/microcoder.c $(FW)/model_data.c $(FW)/prompts.c
	$(CC) $(CFLAGS) -o $@ $^ -lm
clean:
	rm -f sim
