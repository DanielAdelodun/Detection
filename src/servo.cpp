#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <chrono>

#include <signal.h>

#include <pigpiod_if2.h>

#include "servo.hpp"

Gimbal::Gimbal(std::string rpi_hostname, int gpio_pitch, int gpio_roll, int gpio_yaw):
  gpio_pitch(gpio_pitch), gpio_roll(gpio_roll), gpio_yaw(gpio_yaw), rpi_hostname(rpi_hostname)
{
  pi = pigpio_start(rpi_hostname.c_str(), NULL);
  if (pi < 0) {
    fprintf(stderr, "Gimbal: Failed to connect to pigpio daemon\n");
    return;
  }
}

int Gimbal::get_pi() { return pi; }

int Gimbal::set_pitch_pulsewidth(int pitch_pulsewidth) {
  if (pitch_pulsewidth < pitch_min || pitch_pulsewidth > pitch_max)
    return -1;
  printf("Setting %d pitch to %d\n", gpio_pitch, pitch_pulsewidth);
  if (set_servo_pulsewidth(pi, gpio_pitch, pitch_pulsewidth) != 0)
    return -1;
  pitch = pitch_pulsewidth;
  return pitch;
}

int Gimbal::set_roll_pulsewidth(int roll_pulsewidth) {
  if (roll_pulsewidth < roll_min || roll_pulsewidth > roll_max)
    return -1;
  printf("Setting %d roll to %d\n", gpio_roll, roll_pulsewidth);
  if (set_servo_pulsewidth(pi, gpio_roll, roll_pulsewidth) != 0)
    return -1;
  roll = roll_pulsewidth;
  return roll;
}

int Gimbal::set_yaw_pulsewidth(int yaw_pulsewidth) {
  if (yaw_pulsewidth < yaw_min || yaw_pulsewidth > yaw_max)
    return -1;
  printf("Setting %d yaw to %d\n", gpio_yaw, yaw_pulsewidth);
  if (set_servo_pulsewidth(pi, gpio_yaw, yaw_pulsewidth) != 0)
    return -1;
    yaw = yaw_pulsewidth;
    return yaw;
}

int Gimbal::disconnect_pitch() {
  return set_servo_pulsewidth(pi, gpio_pitch, 0);
}

int Gimbal::disconnect_roll() {
  return set_servo_pulsewidth(pi, gpio_roll, 0);
}

int Gimbal::disconnect_yaw() {
  return set_servo_pulsewidth(pi, gpio_yaw, 0);
}

void Gimbal::disconnect() {
  disconnect_yaw(); disconnect_roll(); disconnect_pitch();
  return pigpio_stop(pi);
}

void pid_loop(Gimbal &gimbal, int x_error, int y_error) {

  static std::chrono::_V2::system_clock::time_point time_previous;
  static bool first_time = true;

  // PID Constants
  float Kp = 0.4;
  float Ki = 0.00001;
  float Kd = 0.00000001;

  // microsseconds past
  std::chrono::_V2::system_clock::time_point time_now = std::chrono::high_resolution_clock::now();

  // time difference in microseconds
  float dt = std::chrono::duration_cast<std::chrono::milliseconds>(time_now - time_previous).count();

  // PID Variables
  
  float x_proportional = 0;
  float y_proportional = 0;

  float x_integral = 0;
  float y_integral = 0;

  float x_derivative = 0;
  float y_derivative = 0;

  static float x_previous_error = 0;
  static float y_previous_error = 0;

  float x_output = 0;
  float y_output = 0;

  // PID Loop

  // Proportional
  x_proportional = Kp * x_error;
  y_proportional = Kp * y_error;

  if (first_time) {
    x_previous_error = x_error;
    y_previous_error = y_error;
    dt = 0.1;
    first_time = false;
  }

  // Integral
  x_integral = x_integral * 0.5 + x_error * (dt / 1000) * Ki;
  y_integral = y_integral * 0.5 + y_error * (dt / 1000) * Ki;

  // Derivative
  x_derivative = (x_error - x_previous_error) / (dt / 1000) * Kd;
  y_derivative = (y_error - y_previous_error) / (dt / 1000) * Kd;

  // Total
  x_output = x_proportional + x_integral + x_derivative;
  y_output = y_proportional + y_integral + y_derivative;

  printf("x_error: %d, y_error: %d, x_output: %f, y_output: %f\n", x_error, y_error, x_output, y_output);

  // Update previous values
  x_previous_error = x_error;
  y_previous_error = y_error;

  // Update previous time
  time_previous = time_now;

  // Update gimbal
  printf("Set yaw to %d\n", gimbal.set_yaw_pulsewidth(gimbal.yaw + (int) x_output));
  printf("Set pitch to %d\n", gimbal.set_pitch_pulsewidth(gimbal.pitch + (int) y_output));
}



