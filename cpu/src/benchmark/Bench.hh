#pragma once



#include <chrono>
#include <string>
#include <unordered_map>

namespace bench {
    class Bench {
    public:

        Bench(const Bench&) = delete;
        Bench(const Bench&&) = delete;
        Bench operator=(const Bench&) = delete;

        static void start(const std::string& name, std::chrono::time_point<std::chrono::steady_clock> start);
        static void end(const std::string& name, std::chrono::time_point<std::chrono::steady_clock> end);
        static std::chrono::duration<double> duration(const std::string& name);

    protected:
        Bench() = default;


    private:
        static std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> timers_start;
        static std::unordered_map<std::string, std::chrono::time_point<std::chrono::steady_clock>> timers_end;

    };
}
