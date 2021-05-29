//
// Created by dominique on 5/28/21.
//

#include "Bench.hh"


void bench::Bench::start(const std::string& name, std::chrono::time_point<std::chrono::steady_clock> start) {
    timers_start.insert({ name, start });
}

void bench::Bench::end(const std::string& name, std::chrono::time_point<std::chrono::steady_clock> end) {
    timers_end.insert({ name, end });
}

std::chrono::duration<double> bench::Bench::duration(const std::string &name) {

    auto start = timers_start.at(name);
    auto end = timers_end.at(name);
    return std::chrono::duration<double>(end - start);
}

