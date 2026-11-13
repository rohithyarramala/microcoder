# MicroPython: "MicroCoder field assistant" — an offline code helper that runs
# entirely on an ESP32-S3. A technician describes a small sensor-processing
# function in plain English (over the USB serial REPL); the device generates
# it, verifies it with compile(), shows it, and on confirmation installs it so
# it can be applied to live sensor data immediately — no cloud, no laptop.
#
#   >>> import assistant
#   >>> assistant.run()
#   describe> Write a function to_volts(raw) that converts a 12-bit ADC reading raw to volts with a 3.3 V reference.
#
import time
import ref

INSTALLED = {}


def _fn_name(code):
    first = code.split("\n", 1)[0]
    if first.startswith("def ") and "(" in first:
        return first[4:first.index("(")].strip()
    return None


def make(instr, verbose=True):
    t0 = time.ticks_ms()
    code, n, trace = ref.solve(instr)
    dt = time.ticks_diff(time.ticks_ms(), t0)
    if verbose:
        print("--- generated in %d ms, %d generation(s), %s" % (dt, n, "verified by compile()" if code else "FAILED gate"))
        print(code if code else trace[-1])
    return code


def install(code):
    g = {}
    exec(code, g)
    name = _fn_name(code)
    INSTALLED[name] = g[name]
    return g[name]


def apply_to_adc(fn, pin=1, n=8, period_ms=250):
    """Example: stream a live ADC channel through the generated function."""
    from machine import ADC, Pin
    adc = ADC(Pin(pin))
    adc.atten(ADC.ATTN_11DB)
    for _ in range(n):
        raw = adc.read()  # 12-bit on ESP32
        print(raw, "->", fn(raw))
        time.sleep_ms(period_ms)


def run():
    print("MicroCoder field assistant. Empty line to quit.")
    while True:
        instr = input("describe> ").strip()
        if not instr:
            break
        code = make(instr)
        if code and input("install? [y/N] ").strip().lower() == "y":
            fn = install(code)
            print("installed", _fn_name(code), "- call assistant.INSTALLED[...]")
