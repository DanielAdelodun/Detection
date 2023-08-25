#ifndef SERVO_HPP
#define SERVO_HPP

#include <string>

void pid_loop(Gimbal &gimbal, int x_error, int y_error);

class Gimbal {
public:
    int gpio_pitch;
    int gpio_roll;
    int gpio_yaw;

    Gimbal(std::string = "ariel.local", int = 19, int = 20, int = 21);
    int get_pi();
    int set_pitch_pulsewidth(int);
    int set_roll_pulsewidth(int);
    int set_yaw_pulsewidth(int);
    int disconnect_pitch();
    int disconnect_roll();
    int disconnect_yaw();
    void disconnect();
};

#endif