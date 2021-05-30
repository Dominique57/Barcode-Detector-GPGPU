#pragma once



#include <chrono>
#include <string>
#include <unordered_map>

namespace bench {
    class Bench {
    public:
        using t_steady_chrono = std::chrono::time_point<std::chrono::steady_clock>;
        Bench(const Bench&) = delete;
        Bench(const Bench&&) = delete;
        Bench operator=(const Bench&) = delete;

        // Add a new start time in timers_start mapped with a string.
        static void start(const std::string& name);

        // Add a new end time in timers_end mapped with a string.
        static void end(const std::string& name);

        // The duration computed between the specified start time and specified end time.
        static double duration(const std::string& name, const std::string& type = "seconds");

        static void print(std::ostream& out, const std::string& name, const std::string& type = "seconds");


    protected:
        // Constructor declared protected so that it can't be instantiate.
        Bench() = default;

    private:
        static std::unordered_map<std::string, t_steady_chrono> timers_start;
        static std::unordered_map<std::string, t_steady_chrono> timers_end;
    };
}

