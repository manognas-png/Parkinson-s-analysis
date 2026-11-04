# Insole firmware (Raspberry Pi Pico, Wokwi simulator)

The insole-only Parkinson's gait model running on a simulated microcontroller.
**Research demonstration only, not a diagnostic tool.**

## What is here

| File | Role |
|---|---|
| `lib/gait_model/gait_model.c, .h` | The model in plain C: stride detection, per-stride estimates, end-of-walk classifier. Same code on the Pico and on the cluster. |
| `include/model_params.h` | Model parameters (written by `export_model.py`): 3 × 600 int16 values + 17 floats, about 3.7 kB |
| `include/demo_recordings.h` | Two PhysioNet recordings (GaCo01 healthy, GaPt29 Parkinson's), 6 sensors, 100 Hz |
| `src/main.cpp` | Pico program: replays a recording through the model, OLED + serial output, button, knob, LED |
| `export_model.py` | Trains the model with the device's exact steps, cross-validates it, writes the headers |
| `validate_firmware.py` | Compiles the C code on the cluster and checks it against Python on all 165 recordings |
| `host/run_host.c` | Runs the C model on a PhysioNet file on the cluster |
| `platformio.ini`, `wokwi.toml`, `diagram.json` | Pico build settings and the simulated circuit |
| `../src/firmware_reference.py` | Python reference that mirrors the C code step by step |

## Steps

1. **Export and validate** (SLURM, from the project root):
   `sbatch --account=bioinfmath540f26_class slurm/firmware.sbatch`
2. **Build the Pico firmware** (login node; the first build downloads the Pico compiler, a few hundred MB):
   ```bash
   source .venv/bin/activate
   pip install platformio
   pio run -d firmware
   ```
3. **Simulate**: install the *Wokwi Simulator* extension in VS Code (it asks for a free Wokwi
   licence the first time), open `firmware/diagram.json`, and choose *Wokwi: Start Simulator*.

## Using the simulation

- The Pico replays the healthy recording at real speed; each stride appears on the OLED and in the
  serial monitor with the time the Pico needed for it.
- After about 2 minutes the end-of-walk result appears; the red LED lights if the model output is above 0.5.
- **Button**: switch to the other recording. **Knob**: heel-sensor gain (middle = unchanged). Turning
  the heel down makes the healthy recording look more like Parkinsonian loading.
