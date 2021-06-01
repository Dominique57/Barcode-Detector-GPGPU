//
// Created by dominique on 5/28/21.
//

#include "Bench.hh"

std::unordered_map<std::string, bench::Bench::t_steady_chrono> bench::Bench::timers_start;
std::unordered_map<std::string, bench::Bench::t_steady_chrono> bench::Bench::timers_end;

void bench::Bench::start(const std::string& name) {
    timers_start.insert({ name, std::chrono::steady_clock::now()});
}

void bench::Bench::end(const std::string& name) {
    timers_end.insert({ name, std::chrono::steady_clock::now()});
}

double bench::Bench::duration(const std::string &name, const std::string &type) {
    double duration;
    auto start = timers_start.at(name);
    auto end = timers_end.at(name);
    if (type == "milliseconds") {
        // duration = std::chrono::duration_cast<std::chrono::milliseconds >(end - start).count();
        duration = (1000 * std::chrono::duration<double>(end - start)).count();
    }
    else {
        duration = std::chrono::duration<double>(end - start).count();
    }
    return duration;
}

void bench::Bench::print(std::ostream &out, const std::string &name, const std::string &type) {

    out << "The " << name << " program took: " <<  duration(name, type) << " " << type << std::endl;
}
