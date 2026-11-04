/*
 * Run the firmware's gait model on a PhysioNet trial file, on the cluster (no microcontroller).
 *   ./run_host data/raw/GaCo01_01.txt
 * Prints one line of JSON: strides, features, logit, probability.
 * The input is converted exactly as on the device: sensors L1, L4, L7, R1, R4, R7 as
 * int16 in units of 0.1 N (rounded half to even, like numpy).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "gait_model.h"

static gm_state state; /* large: keep it out of the stack */

int main(int argc, char **argv) {
  if (argc != 2) { fprintf(stderr, "usage: %s trial.txt\n", argv[0]); return 2; }
  FILE *f = fopen(argv[1], "r");
  if (!f) { perror(argv[1]); return 1; }
  const int columns[GM_SENSORS] = {1, 4, 7, 9, 12, 15}; /* L1 L4 L7 R1 R4 R7 (column 0 = time) */
  double row[19];
  int16_t sample[GM_SENSORS];
  gm_init(&state);
  while (1) {
    int read = 0;
    for (int k = 0; k < 19; k++) read += fscanf(f, "%lf", &row[k]) == 1;
    if (read < 19) break;
    for (int c = 0; c < GM_SENSORS; c++) sample[c] = (int16_t)rint(row[columns[c]] * 10);
    gm_push(&state, sample);
  }
  fclose(f);
  gm_result r;
  gm_finish(&state, &r);
  printf("{\"n_samples\": %ld, \"n_strides\": %d, \"n_steady\": %d, \"features\": [", state.n_samples, r.n_strides,
         r.n_steady);
  for (int i = 0; i < GM_N_FEATURES; i++) printf(i ? ", %.17g" : "%.17g", (double)r.features[i]);
  printf("], \"logit\": %.17g, \"probability\": %.17g}\n", (double)r.logit, (double)r.probability);
  return 0;
}
