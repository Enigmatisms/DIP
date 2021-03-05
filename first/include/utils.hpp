#pragma once
#include <chrono>

inline uint64_t getCurrentTime() {
    std::chrono::system_clock clk;
    return clk.now().time_since_epoch().count();
}