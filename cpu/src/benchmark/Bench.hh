#pragma once

#include <chrono>
#include <hash_map>
#include <string>

namespace bench {
    class Bench {
    public:
        Bench(const Bench&) = delete;
        Bench(const Bench&&) = delete;
        Bench operator=(const Bench&) = delete;

    protected:
        Bench() = default;

    private:
        static std::unordered_map<std::string, unsigned> timers;
    };
}
