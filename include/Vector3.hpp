#pragma once

#include <cmath>

struct Vector3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    Vector3() = default;
    Vector3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    Vector3 operator+(const Vector3& rhs) const { return {x + rhs.x, y + rhs.y, z + rhs.z}; }
    Vector3 operator-(const Vector3& rhs) const { return {x - rhs.x, y - rhs.y, z - rhs.z}; }
    Vector3 operator*(double scalar) const { return {x * scalar, y * scalar, z * scalar}; }
    Vector3 operator/(double scalar) const { return {x / scalar, y / scalar, z / scalar}; }

    Vector3& operator+=(const Vector3& rhs) {
        x += rhs.x; y += rhs.y; z += rhs.z;
        return *this;
    }

    Vector3& operator-=(const Vector3& rhs) {
        x -= rhs.x; y -= rhs.y; z -= rhs.z;
        return *this;
    }

    Vector3& operator*=(double scalar) {
        x *= scalar; y *= scalar; z *= scalar;
        return *this;
    }

    double dot(const Vector3& rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }
    double lengthSquared() const { return dot(*this); }
    double length() const { return std::sqrt(lengthSquared()); }
};

inline Vector3 operator*(double scalar, const Vector3& v) { return v * scalar; }
