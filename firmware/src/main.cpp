/*
 * Insole gait demo for the Raspberry Pi Pico (runs in the Wokwi simulator).
 *
 * Replays a real PhysioNet recording (6 insole sensors, 100 Hz) sample by sample, as if read
 * from the sensors, through the same gait model that was validated against Python on the
 * cluster (lib/gait_model). Shows each stride on the OLED, prints it on the serial monitor
 * with the time the Pico spent on it, and gives the model's output at the end of the walk.
 *
 *   Button (GP15): switch between the healthy and the Parkinson's recording
 *   Knob   (GP26): heel-sensor gain, 0-200 % (middle = 100 %, unchanged recording)
 *   LED    (GP16): on when the model output is above 0.5
 *   OLED   (I2C GP4/GP5), serial monitor (UART0 GP0/GP1, 115200 baud)
 *
 * Research demonstration only, not a diagnostic tool.
 */
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Arduino.h>
#include <Wire.h>

#include "demo_recordings.h"
#include "gait_model.h"

const int BUTTON_PIN = 15;
const int LED_PIN = 16;
const int KNOB_PIN = A0;  // GP26
const unsigned long SAMPLE_PERIOD_US = 10000;  // 100 Hz

Adafruit_SSD1306 display(128, 64, &Wire, -1);
static gm_state state;  // about 13 kB: keep it out of the stack

int recording = 0;
long sample_index = 0;
int heel_percent = 100;  // knob position as a whole percentage, 0-200
unsigned long next_sample_us = 0;
unsigned long longest_stride_us = 0;
bool finished = false;
unsigned long finished_at_ms = 0;
unsigned long last_button_ms = 0;

/* "12.34" with the given decimals, without printf %f (some Arduino cores leave it out). */
const char *fixed(float value, int decimals) {
  static char buffers[4][16];
  static int next = 0;
  char *out = buffers[next];
  next = (next + 1) % 4;
  long scale = 1;
  for (int i = 0; i < decimals; i++) scale *= 10;
  long v = lroundf(value * scale);
  const char *sign = v < 0 ? "-" : "";
  if (v < 0) v = -v;
  if (decimals == 0) snprintf(out, 16, "%s%ld", sign, v);
  else snprintf(out, 16, "%s%ld.%0*ld", sign, v / scale, decimals, v % scale);
  return out;
}

void show(const char *line1, const char *line2, const char *line3, const char *line4) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  const char *lines[] = {line1, line2, line3, line4};
  for (int i = 0; i < 4; i++) {
    display.setCursor(0, i * 16);
    display.print(lines[i]);
  }
  display.display();
}

void start_recording(int index) {
  recording = index;
  sample_index = 0;
  longest_stride_us = 0;
  finished = false;
  digitalWrite(LED_PIN, LOW);
  gm_init(&state);
  next_sample_us = micros();
  Serial1.print("\n== Recording ");
  Serial1.print(GM_DEMO[recording].name);
  Serial1.print(" (");
  Serial1.print(GM_DEMO[recording].group);
  Serial1.print("), ");
  Serial1.print(GM_DEMO[recording].n / 100.0, 1);
  Serial1.println(" s ==");
  Serial1.println("stride, time_s, est_speed_m_s, est_TUG_s, deviation, compute_us");
  char line1[24];
  snprintf(line1, sizeof line1, "%s (%s)", GM_DEMO[recording].name, GM_DEMO[recording].group);
  show(line1, "walking...", "", "");
}

void report_stride(unsigned long compute_us) {
  const gm_stride &s = state.last;
  if (compute_us > longest_stride_us) longest_stride_us = compute_us;
  Serial1.print(state.n_strides);
  Serial1.print(", ");
  Serial1.print(s.stride_time, 2);
  Serial1.print(", ");
  Serial1.print(s.speed, 3);
  Serial1.print(", ");
  Serial1.print(exp(s.log_tug), 2);
  Serial1.print(", ");
  Serial1.print(s.deviation, 0);
  Serial1.print(", ");
  Serial1.println(compute_us);

  char line1[24], line2[24], line3[24], line4[24];
  snprintf(line1, sizeof line1, "%s (%s)", GM_DEMO[recording].name, GM_DEMO[recording].group);
  snprintf(line2, sizeof line2, "stride %d  %s s", state.n_strides, fixed(s.stride_time, 2));
  snprintf(line3, sizeof line3, "speed est %s m/s", fixed(s.speed, 2));
  snprintf(line4, sizeof line4, "heel gain %d%%  %luus", heel_percent, compute_us);
  show(line1, line2, line3, line4);
}

void finish_recording() {
  const unsigned long t0 = micros();
  gm_result r;
  gm_finish(&state, &r);
  const unsigned long finish_us = micros() - t0;
  finished = true;
  finished_at_ms = millis();
  digitalWrite(LED_PIN, r.probability > 0.5f ? HIGH : LOW);

  const char *names[] = {"est_speed", "est_log_tug", "template_dev", "energy_mode1", "effective_modes",
                         "heel_forefoot"};
  Serial1.print("strides: ");
  Serial1.print(r.n_strides);
  Serial1.print(", steady: ");
  Serial1.println(r.n_steady);
  for (int i = 0; i < GM_N_FEATURES; i++) {
    Serial1.print(names[i]);
    Serial1.print(" = ");
    Serial1.println(r.features[i], 4);
  }
  Serial1.print("model output (probability of a Parkinsonian pattern) = ");
  Serial1.println(r.probability, 3);
  Serial1.print("compute time: longest stride ");
  Serial1.print(longest_stride_us);
  Serial1.print(" us, end of walk ");
  Serial1.print(finish_us);
  Serial1.println(" us");
  Serial1.println("Research demonstration only, not a diagnostic tool.");

  char line1[24], line2[24], line3[24], line4[24];
  snprintf(line1, sizeof line1, "%s done", GM_DEMO[recording].name);
  snprintf(line2, sizeof line2, "%d strides, %d steady", r.n_strides, r.n_steady);
  snprintf(line3, sizeof line3, "PD output %s", fixed(r.probability, 2));
  snprintf(line4, sizeof line4, "%s", r.probability > 0.5f ? "pattern: PD-like" : "pattern: typical");
  show(line1, line2, line3, line4);
}

void setup() {
  Serial1.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  Serial1.println("Insole gait demo. Research demonstration only, not a diagnostic tool.");
  start_recording(0);
}

void loop() {
  if (digitalRead(BUTTON_PIN) == LOW && millis() - last_button_ms > 400) {
    last_button_ms = millis();
    start_recording((recording + 1) % GM_DEMO_COUNT);
    return;
  }
  if (finished) {
    if (millis() - finished_at_ms > 8000) start_recording(recording);  // replay after 8 s
    return;
  }
  if ((long)(micros() - next_sample_us) < 0) return;
  next_sample_us += SAMPLE_PERIOD_US;

  if (sample_index % 10 == 0) heel_percent = (analogRead(KNOB_PIN) * 200L + 511) / 1023;  // middle = exactly 100
  int16_t sample[GM_SENSORS];
  for (int c = 0; c < GM_SENSORS; c++) sample[c] = GM_DEMO[recording].data[sample_index][c];
  const int heels[2] = {0, 3};  // heel sensors L1 and R1
  for (int h = 0; h < 2; h++) {
    const long scaled = (long)sample[heels[h]] * heel_percent;  // integer maths: 100 % leaves it unchanged
    const long rounded = (scaled + (scaled >= 0 ? 50 : -50)) / 100;
    sample[heels[h]] = (int16_t)constrain(rounded, -32768L, 32767L);
  }
  const unsigned long t0 = micros();
  const int stride_done = gm_push(&state, sample);
  const unsigned long compute_us = micros() - t0;
  if (stride_done) report_stride(compute_us);

  if (++sample_index >= GM_DEMO[recording].n) finish_recording();
}
