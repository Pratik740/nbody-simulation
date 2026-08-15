#include <cmath>

#include "Simulation.hpp"

int main() {
    const double G = 1.0;
    const double softening = 1e-3;
    Simulation sim(G, softening);

    // Central "star" at rest at the origin.
    sim.addParticle(Particle(Vector3(0.0, 0.0, 0.0), Vector3(0.0, 0.0, 0.0), 1000.0));

    // A "planet" on a circular orbit: v = sqrt(G * M / r) balances gravitational
    // pull with the centripetal acceleration needed to hold a circular path.
    const double r = 10.0;
    const double orbitalSpeed = std::sqrt(G * 1000.0 / r);
    sim.addParticle(Particle(Vector3(r, 0.0, 0.0), Vector3(0.0, orbitalSpeed, 0.0), 1.0));

    // A lighter third body on an inclined path, making this a genuine n-body case
    // rather than a two-body problem with a spectator.
    sim.addParticle(Particle(Vector3(0.0, 20.0, 5.0), Vector3(-3.0, 0.0, 0.0), 0.5));

    const double dt = 0.01;
    const int steps = 2000;
    const int printInterval = 100;

    for (int step = 0; step <= steps; ++step) {
        if (step % printInterval == 0) {
            sim.printState(step * dt);
        }
        sim.step(dt);
    }

    return 0;
}