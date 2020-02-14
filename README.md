# particle-filter

A C11 implementation of a particle filter.

Initial samples are drawn from a multivariate Gaussian distribution with
user-provided population mean and covariance. Noise vectors for the prediction
step, used as inputs to the process model, are drawn from a zero-mean
multivariate Gaussian distribution with user-provided covariance. The process
noise covariance may be a function of the action. Conversely, the measurement
model may be entirely arbitrary, as libpf only uses the probability density
function of the posterior.

## Example

See [`odometry.c`](examples/odometry.c) for an example using the 2D velocity
motion model from Probabilistic Robotics and a single landmark measurement
model.
