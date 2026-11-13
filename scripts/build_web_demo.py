"""Build the self-contained browser playground (app/web/playground.html) from a checkpoint."""
import base64, json, sys
sys.path.insert(0, ".")
import torch
from microcoder.export import export_model, export_tok

ck = torch.load(sys.argv[1])
mb, tb = export_model(ck), export_tok(ck["args"]["tok"])
ex = [
    dict(label="ADC to volts", text="Write a function adc_volts(raw) that converts a 12-bit ADC reading raw to volts with a 3.3 V reference."),
    dict(label="Moving average", text="Write a function smooth(samples) that returns the average of the last 4 values in the list samples."),
    dict(label="Hysteresis", text="Define thermostat(on, temp) implementing a hysteresis switch: turn on above 30, off below 25, else return on unchanged."),
    dict(label="XOR checksum", text="Write a function frame_xor(packet) that returns the XOR of all bytes in the list packet."),
    dict(label="Test a bit", text="Define flag_on(status) returning whether bit number 3 of the register status is 1."),
    dict(label="Celsius to F", text="Create to_fahrenheit(temp) to convert degrees Celsius temp into degrees Fahrenheit."),
]
html = open("app/web/template.html").read()
html = html.replace("__ENGINE_JS__", open("app/web/engine.js").read())
html = html.replace("__MODEL_B64__", base64.b64encode(mb).decode()).replace("__TOK_B64__", base64.b64encode(tb).decode())
html = html.replace("__EXAMPLES__", json.dumps(ex))
open("app/web/playground.html", "w").write(html)
print("wrote app/web/playground.html", len(html), "bytes")
