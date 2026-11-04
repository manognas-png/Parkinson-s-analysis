/*
 * Insole gait model: streaming stride detection + the insole-only PD classifier.
 * Plain C99, no Arduino code, so the same file runs on the Pico and on the cluster
 * (where it is checked against src/firmware_reference.py).
 *
 * Usage: gm_init(&s); for each sample: if (gm_push(&s, sample)) { s.last is a new stride };
 *        at the end: gm_finish(&s, &result).
 * Samples: 6 sensors (L1, L4, L7, R1, R4, R7) as int16 in units of 0.1 N, at 100 Hz.
 */
#ifndef GAIT_MODEL_H
#define GAIT_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GM_USE_FLOAT
typedef float gm_real; /* Pico build: single precision (no FPU, so smaller and faster) */
#else
typedef double gm_real; /* cluster build: double, to match Python exactly */
#endif

#define GM_SENSORS 6
#define GM_POINTS 100
#define GM_MAX_STRIDE_SAMPLES 400 /* 4 s; longer strides are counted but not analysed */
#define GM_MAX_STRIDES 400        /* about 7 minutes of walking */
#define GM_N_FEATURES 6

typedef struct {
  int samples;         /* stride length in samples (used for the steady test, in exact integers) */
  gm_real stride_time; /* seconds */
  gm_real speed;       /* estimated walking speed, m/s (NaN if not analysed) */
  gm_real log_tug;     /* estimated log TUG time (fall-risk proxy only) */
  gm_real deviation;   /* mean squared distance to the healthy template, N^2 */
} gm_stride;

typedef struct {
  int16_t buffer[GM_MAX_STRIDE_SAMPLES + 1][GM_SENSORS]; /* samples since the last heel strike */
  int buffer_length;
  int overflow;
  long n_samples;
  long previous_strike; /* sample index of the last heel strike, -1 if none yet */
  int off_run;          /* consecutive samples with left force <= 20 N */
  double sum[GM_SENSORS];
  double cross[GM_SENSORS][GM_SENSORS]; /* whole-recording sums for the covariance */
  gm_stride strides[GM_MAX_STRIDES];
  int n_strides;
  gm_stride last;
} gm_state;

typedef struct {
  int n_strides;
  int n_steady;
  gm_real features[GM_N_FEATURES]; /* est_speed, est_log_tug, template_dev, energy_mode1,
                                      effective_modes, heel_forefoot (before imputation) */
  gm_real logit;
  gm_real probability; /* model output: probability of a Parkinsonian gait pattern */
} gm_result;

void gm_init(gm_state *s);
int gm_push(gm_state *s, const int16_t sample[GM_SENSORS]); /* returns 1 when a stride completes */
void gm_finish(const gm_state *s, gm_result *r);

#ifdef __cplusplus
}
#endif

#endif
