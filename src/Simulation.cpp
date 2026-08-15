#include "Simulation.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>

Simulation::Simulation(double gravitationalConstant, double softeningLength)
    : G_(gravitationalConstant), softening2_(softeningLength * softeningLength) {}

void Simulation::addParticle(const Particle& particle) {
    particles_.push_back(particle);
}

void Simulation::computeAccelerations() {
    for (auto& p : particles_) {
        p.acceleration = Vector3();
    }

    const std::size_t n = particles_.size();
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            Vector3 delta = particles_[j].position - particles_[i].position;
            // Softening keeps the force finite as separation -> 0, which matters
            // once close encounters happen (unavoidable with only a few bodies).
            double distSquared = delta.lengthSquared() + softening2_;
            double dist = std::sqrt(distSquared);
            double forceOverDistTimesMass = G_ / (distSquared * dist);

            particles_[i].acceleration += delta * (forceOverDistTimesMass * particles_[j].mass);
            particles_[j].acceleration -= delta * (forceOverDistTimesMass * particles_[i].mass);
        }
    }
}

// Velocity Verlet (symplectic): keeps orbital energy bounded over long
// integrations instead of secularly drifting like explicit Euler would.
void Simulation::step(double dt) {

    // Calculate new position = old position + curr velocity & curr acceleration
    for (auto& p : particles_) {
        p.position += p.velocity * dt + p.acceleration * (0.5 * dt * dt);
        // s = ut + 0.5 * a * t^2;
    }

    std::vector<Vector3> previousAcceleration;
    previousAcceleration.reserve(particles_.size());
    for (const auto& p : particles_) {
        previousAcceleration.push_back(p.acceleration);
    }

    computeAccelerations();

    // Update velocities using an average of old acc and new acc
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        particles_[i].velocity += (previousAcceleration[i] + particles_[i].acceleration) * (0.5 * dt);
        // v += (a1 + a2)/2 * t   :::   v = u + a' * t 
    }
}

// The issue with the implicit euler position which is
//  i. calculate new velocity based on new accl ie. v = v_old + a*dt;
// ii. calculate new postion = old position + new velocity*dt 
// The issue here is, it assumes that the acceleration remains constant within the dt time interval, but since the space is 
// continuously moving, that assumption is wrong (since with every movement, F changes -> acc CHANGES)
// Accumulated error per dt time step will leads to orbits (ie. positions) artifically gaining energy, causing particles to 
// spiral outwards and move away. 

// Velocity Verlet calculates the exact midpoint of the velocity between the frames using both the current acceleration and the next acceleration.


void Simulation::printState(double time) const {
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < particles_.size(); ++i) {
        const auto& p = particles_[i];
        std::cout << "t=" << time
                  << " p" << i
                  << " pos=(" << p.position.x << ", " << p.position.y << ", " << p.position.z << ")"
                  << " vel=(" << p.velocity.x << ", " << p.velocity.y << ", " << p.velocity.z << ")"
                  << '\n';
    }
}