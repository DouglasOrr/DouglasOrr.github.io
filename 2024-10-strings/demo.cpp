#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_LIBCPP_VERSION)
const char* STDLIB_NAME = "libc++";
#elif defined(_GLIBCXX_RELEASE)
const char* STDLIB_NAME = "libstdc++";
#endif

#if defined(__clang__)
const char* COMPILER_NAME = "clang";
#elif defined(__GNUG__)
const char* COMPILER_NAME = "gcc";
#endif

template <class T>
void show(const std::string& label, T&& value) {
    std::cerr << std::setw(44) << label << ": " << value << "\n";
}
#define SHOW(expr) show(#expr, expr);

bool isSmallString(const std::string& s) {
    auto distance = std::distance((const char*)(&s), &s[0]);
    return (0 <= distance) && (distance < long(sizeof(s)));
}

struct Timer {
    using clock = std::chrono::high_resolution_clock;
    clock::time_point start;
    Timer() : start(clock::now()) {}
    double measure() const {
        return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - start)
            .count();
    }
};

struct PerfTestString {
    static inline const char* const Name = "std::string";
    std::unordered_map<std::string, size_t> map;

    size_t measure() const {
        auto size = map.bucket_count() * sizeof(*map.begin());
        size +=
            (!isSmallString(map.begin()->first)) * map.size() * (map.begin()->first.capacity() + 1);
        return size;
    }
    __attribute_noinline__ void fill(const std::vector<std::string>& values) {
        for (auto i = 0u; i < values.size(); ++i) {
            map[values[i]] = i;
        }
    }
    __attribute_noinline__ size_t query(const std::vector<std::string>& values) {
        auto sum = 0u;
        for (auto& value : values) {
            sum += map.find(value)->second;
        }
        return sum;
    }
};
struct PerfTestStringView {
    static inline const char* const Name = "std::string_view";
    std::unordered_map<std::string_view, size_t> map;

    size_t measure() const { return map.bucket_count() * sizeof(*map.begin()); }
    __attribute_noinline__ void fill(const std::vector<std::string>& values) {
        for (auto i = 0u; i < values.size(); ++i) {
            map[values[i]] = i;
        }
    }
    __attribute_noinline__ size_t query(const std::vector<std::string>& values) {
        auto sum = 0u;
        for (auto& value : values) {
            sum += map.find(value)->second;
        }
        return sum;
    }
};
struct PerfTestStringViewWithPool {
    static inline const char* const Name = "std::string_view (pool)";
    std::vector<char> data;
    std::unordered_map<std::string_view, size_t> map;

    size_t measure() const { return map.bucket_count() * sizeof(*map.begin()) + data.capacity(); }
    __attribute_noinline__ void fill(const std::vector<std::string>& values) {
        auto length = values[0].size();
        data.resize(values.size() * length);
        for (auto i = 0u; i < values.size(); ++i) {
            std::copy(values[i].begin(), values[i].end(), data.data() + i * length);
            map[std::string_view(data.data() + i * length, length)] = i;
        }
    }
    __attribute_noinline__ size_t query(const std::vector<std::string>& values) {
        auto sum = 0u;
        for (auto& value : values) {
            sum += map.find(value)->second;
        }
        return sum;
    }
};
struct PerfTestReferenceWrapper {
    static inline const char* const Name = "std::reference_wrapper";
    std::unordered_map<std::reference_wrapper<const std::string>,
                       size_t,
                       std::hash<std::string>,
                       std::equal_to<std::string>>
        map;

    size_t measure() const { return map.bucket_count() * sizeof(*map.begin()); }
    __attribute_noinline__ void fill(const std::vector<std::string>& values) {
        for (auto i = 0u; i < values.size(); ++i) {
            map[values[i]] = i;
        }
    }
    __attribute_noinline__ size_t query(const std::vector<std::string>& values) {
        auto sum = 0u;
        for (auto& value : values) {
            sum += map.find(value)->second;
        }
        return sum;
    }
};

template <class T>
void runTest(const std::vector<std::string>& values,
             const std::vector<std::string>& queryValues,
             std::ostream& out) {
    T test;
    auto timer = Timer();
    test.fill(values);
    auto fillTime = timer.measure();

    timer = Timer();
    auto result = test.query(queryValues);
    auto queryTime = timer.measure();

    out << "{"                                          //
        << "\"compiler\":\"" << COMPILER_NAME << "\""   //
        << ",\"stdlib\":\"" << STDLIB_NAME << "\""      //
        << ",\"implementation\":\"" << T::Name << "\""  //
        << ",\"count\":" << values.size()               //
        << ",\"length\":" << values[0].size()           //
        << ",\"t_fill\":" << fillTime                   //
        << ",\"t_query\":" << queryTime                 //
        << ",\"map_size\":" << test.measure()           //
        << ",\"_result\":" << result                    //
        << "}\n"
        << std::flush;
}

void testPerformance(const std::string& path) {
    std::ofstream out(path, std::ios_base::app);
    for (auto count : std::vector<size_t>({1 << 16, 1 << 20, 1 << 24})) {
        auto reps = (1 << 24) / count;
        for (auto logLength = 2.0; logLength <= 6; logLength += 0.5) {
            auto length = size_t(std::round(std::pow(2.0, logLength)));
            std::cerr << count << " * " << length << " (" << reps << ")" << "\n";

            // Run the test
            std::vector<std::string> values(count);
            for (auto i = 0u; i < count; ++i) {
                values[i].resize(length, '#');
                *reinterpret_cast<uint32_t*>(values[i].data() + length - sizeof(uint32_t)) = i;
            }
            std::vector<std::string> queryValues(values);
            for (auto i = 0u; i < reps; ++i) {
                runTest<PerfTestString>(values, queryValues, out);
                runTest<PerfTestStringView>(values, queryValues, out);
                runTest<PerfTestStringViewWithPool>(values, queryValues, out);
                runTest<PerfTestReferenceWrapper>(values, queryValues, out);
            }
        }
    }
}

int main(int argc, char** argv) {
    std::cerr << "### SIZES ###\n";
    SHOW(sizeof(std::string));
    SHOW(sizeof(std::string_view));
    SHOW(sizeof(std::reference_wrapper<std::string>));

    std::cerr << "\n### SMALL STRING OPTIMISATION ###\n";
    std::vector<std::string> strings(32);
    for (auto i = 0u; i < strings.size(); ++i) {
        strings[i].resize(i, '#');
    }
    for (auto i = 0u; i < strings.size(); ++i) {
        std::cerr << std::dec << std::setw(4) << i << " "
                  << (isSmallString(strings[i]) ? "yes" : "no") << "\n";
    }

    if (argc >= 2) {
        std::cerr << "\n### PERFORMANCE ###\n";
        std::cerr << "Testing " << STDLIB_NAME << " with " << COMPILER_NAME << " -> " << argv[1]
                  << "\n";
        testPerformance(argv[1]);
    } else {
        std::cerr << "\n### SKIP (PERFORMANCE) ###\n";
    }

    return 0;
}
