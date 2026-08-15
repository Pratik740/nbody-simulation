#pragma once

#include <vector>

#include "Particle.hpp"

class Simulation {
public:
    Simulation(double gravitationalConstant, double softeningLength);

    void addParticle(const Particle& particle);
    void step(double dt);
    void printState(double time) const;

    const std::vector<Particle>& particles() const { return particles_; }

private:
    void computeAccelerations();

    std::vector<Particle> particles_;
    double G_;
    double softening2_;
};