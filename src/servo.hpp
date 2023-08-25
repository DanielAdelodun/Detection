#ifndef SERVO_HPP
#define SERVO_HPP

#include <string>

class Gimbal {
private:
  std::string rpi_hostname;
  int pi;

  int pitch_min = 1000, pitch_max = 2000;
  int roll_min = 1000, roll_max = 2000;
  int yaw_min = 1000, yaw_max = 2000;

  int gpio_pitch;
  int gpio_roll;
  int gpio_yaw;

public:
  int pitch = 1500;
  int roll = 1500;
  int yaw = 1500;

  Gimbal(
    std::string rpi_hostname = "ariel.local", 
    int gpio_pitch = 21, 
    int gpio_roll = 20, 
    int gpio_yaw = 19
  );
  int get_pi();
  int set_pitch_pulsewidth(int);
  int set_roll_pulsewidth(int);
  int set_yaw_pulsewidth(int);
  int disconnect_pitch();
  int disconnect_roll();
  int disconnect_yaw();
  void disconnect();
};

void pid_loop(Gimbal &gimbal, int x_error, int y_error);

#endif