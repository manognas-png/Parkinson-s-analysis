/*
 * Insole gait model (see gait_model.h). Mirrors src/firmware_reference.py step by step.
 * Model parameters come from model_params.h, written by firmware/export_model.py: the three
 * 600-value arrays (ridge weights for speed and TUG, template) as int16 with one scale each
 * (value = int16 * scale); the 15 remaining numbers (log-duration weights, intercepts,
 * logistic weights and imputation medians) as floating point, since their sizes differ widely.
 */
#include "gait_model.h"

#include <math.h>
#include <string.h>

#include "model_params.h"

#define THRESHOLD_N 20.0
#define MIN_SWING 10
#define FS 100.0
#define STEADY_WINDOW 7
#define STEADY_MIN_RUN 10
#define STEADY_TRIM 2

void gm_init(gm_state *s) {
  memset(s, 0, sizeof *s);
  s->previous_strike = -1;
}

/* ---- One stride: resample to 100 points, then the two ridge models and the template ---- */

static void analyse_stride(gm_state *s, int d, gm_stride *out) {
  gm_real speed = (gm_real)GM_SPEED_B;
  gm_real log_tug = (gm_real)GM_TUG_B;
  gm_real deviation = 0;
  const gm_real step = (gm_real)d / GM_POINTS;
  for (int c = 0; c < GM_SENSORS; c++) {
    for (int k = 0; k < GM_POINTS; k++) {
      /* Linear interpolation at k * step samples after the heel strike (numpy interp) */
      const gm_real x = k * step;
      int j = (int)x;
      if (j > d - 1) j = d - 1;
      const gm_real f0 = s->buffer[j][c] / (gm_real)10.0;
      const gm_real f1 = s->buffer[j + 1][c] / (gm_real)10.0;
      const gm_real value = (f1 - f0) * (x - j) + f0;
      const int i = c * GM_POINTS + k;
      speed += (gm_real)GM_SPEED_W[i] * (gm_real)GM_SPEED_W_SCALE * value;
      log_tug += (gm_real)GM_TUG_W[i] * (gm_real)GM_TUG_W_SCALE * value;
      const gm_real diff = value - (gm_real)GM_TEMPLATE[i] * (gm_real)GM_TEMPLATE_SCALE;
      deviation += diff * diff;
    }
  }
  const gm_real log_duration = (gm_real)log(d / FS);
  speed += (gm_real)GM_SPEED_LOGDUR_W * log_duration;
  log_tug += (gm_real)GM_TUG_LOGDUR_W * log_duration;
  out->speed = speed;
  out->log_tug = log_tug;
  out->deviation = deviation / (GM_SENSORS * GM_POINTS);
}

/* ---- Streaming input ---- */

int gm_push(gm_state *s, const int16_t sample[GM_SENSORS]) {
  double x[GM_SENSORS];
  for (int c = 0; c < GM_SENSORS; c++) x[c] = sample[c] / 10.0;
  for (int a = 0; a < GM_SENSORS; a++) {
    s->sum[a] += x[a];
    for (int b = 0; b < GM_SENSORS; b++) s->cross[a][b] += x[a] * x[b];
  }

  /* Heel strike: left force rises above 20 N after at least 10 samples at or below 20 N */
  const int off = (x[0] + x[1] + x[2]) <= THRESHOLD_N;
  const int strike = !off && s->off_run >= MIN_SWING;
  s->off_run = off ? s->off_run + 1 : 0;

  int completed = 0;
  if (s->previous_strike >= 0) {
    if (s->buffer_length <= GM_MAX_STRIDE_SAMPLES) {
      memcpy(s->buffer[s->buffer_length], sample, sizeof s->buffer[0]);
      s->buffer_length++;
    } else {
      s->overflow = 1;
    }
  }
  if (strike) {
    if (s->previous_strike >= 0 && s->n_strides < GM_MAX_STRIDES) {
      const int d = (int)(s->n_samples - s->previous_strike);
      gm_stride stride;
      stride.samples = d;
      stride.stride_time = d / (gm_real)FS;
      if (s->overflow || d > GM_MAX_STRIDE_SAMPLES) {
        stride.speed = stride.log_tug = stride.deviation = (gm_real)NAN;
      } else {
        analyse_stride(s, d, &stride);
      }
      s->strides[s->n_strides++] = stride;
      s->last = stride;
      completed = 1;
    }
    s->previous_strike = s->n_samples; /* new stride starts with this sample */
    memcpy(s->buffer[0], sample, sizeof s->buffer[0]);
    s->buffer_length = 1;
    s->overflow = 0;
  }
  s->n_samples++;
  return completed;
}

/* ---- End of the recording ---- */

static void sort_real(gm_real *v, int n) { /* insertion sort: n is small */
  for (int i = 1; i < n; i++) {
    gm_real key = v[i];
    int j = i - 1;
    while (j >= 0 && v[j] > key) { v[j + 1] = v[j]; j--; }
    v[j + 1] = key;
  }
}

static gm_real median(gm_real *v, int n) {
  if (n == 0) return (gm_real)NAN;
  sort_real(v, n);
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}

/* Eigenvalues of a symmetric 6 x 6 matrix (cyclic Jacobi rotations), in place on a copy. */
static void eigenvalues(double m[GM_SENSORS][GM_SENSORS], double out[GM_SENSORS]) {
  for (int sweep = 0; sweep < 100; sweep++) {
    double off = 0, total = 0;
    for (int p = 0; p < GM_SENSORS; p++)
      for (int q = 0; q < GM_SENSORS; q++) { total += m[p][q] * m[p][q]; if (p != q) off += m[p][q] * m[p][q]; }
    if (off <= 1e-30 * total) break;
    for (int p = 0; p < GM_SENSORS - 1; p++) {
      for (int q = p + 1; q < GM_SENSORS; q++) {
        if (m[p][q] == 0) continue;
        const double theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const double t = (theta >= 0 ? 1 : -1) / (fabs(theta) + sqrt(theta * theta + 1));
        const double c = 1 / sqrt(t * t + 1), sn = t * c;
        for (int k = 0; k < GM_SENSORS; k++) { /* rotate columns p, q */
          const double a = m[k][p], b = m[k][q];
          m[k][p] = c * a - sn * b;
          m[k][q] = sn * a + c * b;
        }
        for (int k = 0; k < GM_SENSORS; k++) { /* rotate rows p, q */
          const double a = m[p][k], b = m[q][k];
          m[p][k] = c * a - sn * b;
          m[q][k] = sn * a + c * b;
        }
      }
    }
  }
  for (int i = 0; i < GM_SENSORS; i++) out[i] = m[i][i];
}

void gm_finish(const gm_state *s, gm_result *r) {
  static gm_real speeds[GM_MAX_STRIDES], tugs[GM_MAX_STRIDES], devs[GM_MAX_STRIDES];
  const int n = s->n_strides;
  int steady[GM_MAX_STRIDES] = {0}, mask[GM_MAX_STRIDES] = {0};

  /* Steady strides: within 10 % of the rolling median of 7, in runs of >= 10, minus 2 at each end.
     Done in exact integers (stride lengths in samples): |d - m| <= m / 10  <=>  10 |2d - 2m| <= 2m,
     and 2m is an integer, so boundary cases do not depend on floating-point rounding. */
  for (int i = 0; i < n; i++) {
    int window[STEADY_WINDOW];
    int m = 0;
    for (int k = i - STEADY_WINDOW / 2; k <= i + STEADY_WINDOW / 2; k++)
      if (k >= 0 && k < n) window[m++] = s->strides[k].samples;
    for (int a = 1; a < m; a++) { /* insertion sort */
      int key = window[a], b = a - 1;
      while (b >= 0 && window[b] > key) { window[b + 1] = window[b]; b--; }
      window[b + 1] = key;
    }
    const long twice_median = m % 2 ? 2L * window[m / 2] : (long)window[m / 2 - 1] + window[m / 2];
    const long gap = 2L * s->strides[i].samples - twice_median;
    steady[i] = 10 * (gap < 0 ? -gap : gap) <= twice_median;
  }
  for (int start = 0; start < n;) {
    if (!steady[start]) { start++; continue; }
    int end = start;
    while (end < n && steady[end]) end++;
    if (end - start >= STEADY_MIN_RUN)
      for (int k = start + STEADY_TRIM; k < end - STEADY_TRIM; k++) mask[k] = 1;
    start = end;
  }

  int kept = 0, n_steady = 0;
  for (int i = 0; i < n; i++) {
    n_steady += mask[i];
    if (mask[i] && !isnan(s->strides[i].speed)) {
      speeds[kept] = s->strides[i].speed;
      tugs[kept] = s->strides[i].log_tug;
      devs[kept] = s->strides[i].deviation;
      kept++;
    }
  }
  r->n_strides = n;
  r->n_steady = n_steady;
  r->features[0] = median(speeds, kept);
  r->features[1] = median(tugs, kept);
  r->features[2] = median(devs, kept);

  /* Gait features from the whole-recording covariance of the 6 sensors */
  double cov[GM_SENSORS][GM_SENSORS], eig[GM_SENSORS];
  const double count = (double)s->n_samples;
  for (int a = 0; a < GM_SENSORS; a++)
    for (int b = 0; b < GM_SENSORS; b++) cov[a][b] = s->cross[a][b] - s->sum[a] * s->sum[b] / count;
  double coupling = 0;
  const int pairs[4][2] = {{0, 1}, {0, 2}, {3, 4}, {3, 5}};
  for (int k = 0; k < 4; k++) {
    const int i = pairs[k][0], j = pairs[k][1];
    coupling += fabs(cov[i][j]) / sqrt(cov[i][i] * cov[j][j]);
  }
  eigenvalues(cov, eig);
  double total = 0, largest = 0, squares = 0;
  for (int i = 0; i < GM_SENSORS; i++) { if (eig[i] < 0) eig[i] = 0; total += eig[i]; if (eig[i] > largest) largest = eig[i]; }
  for (int i = 0; i < GM_SENSORS; i++) squares += (eig[i] / total) * (eig[i] / total);
  r->features[3] = (gm_real)(largest / total);
  r->features[4] = (gm_real)(1 / squares);
  r->features[5] = (gm_real)(coupling / 4);

  /* Logistic regression (scaler folded into the weights; missing features -> training medians) */
  gm_real logit = (gm_real)GM_LOGIT_B;
  for (int i = 0; i < GM_N_FEATURES; i++) {
    gm_real v = r->features[i];
    if (isnan(v)) v = (gm_real)GM_MEDIANS[i];
    logit += (gm_real)GM_LOGIT_W[i] * v;
  }
  r->logit = logit;
  r->probability = (gm_real)(1 / (1 + exp(-logit)));
}
