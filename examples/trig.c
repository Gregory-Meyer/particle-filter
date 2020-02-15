#include "trig.h"

double wrap_to_pi(double angle) {
  while (angle >= PI) {
    angle -= 2.0 * PI;
  }

  while (angle < -PI) {
    angle += 2.0 * PI;
  }

  return angle;
}

double wrap_to_2pi(double angle) {
  while (angle >= 2.0 * PI) {
    angle -= 2.0 * PI;
  }

  while (angle < 0.0) {
    angle += 2.0 * PI;
  }

  return angle;
}
