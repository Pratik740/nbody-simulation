#pragma once

#include "Vector3.hpp"

struct Particle {
    Vector3 position;
    Vector3 velocity;
    Vector3 acceleration;
    double mass;

    Particle(const Vector3& position_, const Vector3& velocity_, double mass_)
        : position(position_), velocity(velocity_), acceleration(), mass(mass_) {}
};
