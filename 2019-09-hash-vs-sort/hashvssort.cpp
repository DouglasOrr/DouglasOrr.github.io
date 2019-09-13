#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <iterator>
#include <random>
#include <unordered_set>
#include <vector>

namespace {

    ////////////////////////////////////////////////////////////////////////////////
    // Core

    template<class T>
    std::vector<T> hash_unique(const std::vector<T>& items) {
        std::unordered_set<T> unique(items.begin(), items.end());
        return std::vector<T>(unique.begin(), unique.end());
    }

    template<class T>
    std::vector<T> sort_unique(const std::vector<T>& items) {
        auto results = items;
        std::sort(results.begin(), results.end());
        auto n = std::unique(results.begin(), results.end()) - results.begin();
        results.resize(n);
        return results;
    }

    template<class T>
    std::vector<T> custom_hash_unique(const std::vector<T>& items) {
        const auto nBuckets = std::max(size_t(1), items.size() / 16);
        const std::hash<T> hash;

        // Compute space in the hash table
        std::vector<size_t> bucketIndices(nBuckets, 0);
        for (const auto& item : items) {
            ++bucketIndices[hash(item) % nBuckets];
        }
        auto offset = 0;
        for (size_t i = 0; i < nBuckets; ++i) {
            auto count = bucketIndices[i];
            bucketIndices[i] = offset;
            offset += count;
        }
        const auto tableSize = offset;

        // Build the hash table of unique items
        std::vector<size_t> bucketCounts(nBuckets, 0);
        std::vector<T> table(tableSize);
        for (const auto& item : items) {
            auto index = hash(item) % nBuckets;
            auto begin = table.begin() + bucketIndices[index];
            auto end = begin + bucketCounts[index];
            if (std::find(begin, end, item) == end) {
                *end = item;
                ++bucketCounts[index];
            }
        }

        // Make the unique items contiguous
        auto dest = table.begin() + bucketCounts[0];
        for (size_t srcBucket = 1; srcBucket < nBuckets; ++srcBucket) {
            auto srcCount = bucketCounts[srcBucket];
            auto srcBegin = table.begin() + bucketIndices[srcBucket];
            if (srcBegin != dest) {
                std::move(srcBegin, srcBegin + srcCount, dest);
            }
            dest += srcCount;
        }
        table.resize(dest - table.begin());
        return table;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Measurement

    typedef std::chrono::steady_clock timing_clock;

    template<class T, class Rand>
    std::vector<T> generateTestItems(size_t nTotal, size_t nUnique, Rand& engine) {
        assert(0 < nUnique && nUnique <= nTotal);
        auto chooseNew = std::uniform_int_distribution<T>(std::numeric_limits<T>::min(),
                                                          std::numeric_limits<T>::max());
        std::unordered_set<T> unique;
        while (unique.size() < nUnique) {
            unique.insert(chooseNew(engine));
        }

        std::vector<T> items(nTotal);
        std::copy(unique.begin(), unique.end(), items.begin());
        auto chooseExisting = std::uniform_int_distribution<size_t>(0, nUnique - 1);
        std::generate(items.begin() + nUnique, items.end(), [&]() {
                return items[chooseExisting(engine)];
            });
        std::shuffle(items.begin(), items.end(), engine);
        return items;
    }

    template<class T>
    struct Element {
        typedef T value_type;
        value_type value;

        static uint64_t s_countDefaultConstruct;
        static uint64_t s_countCopyConstruct;
        static uint64_t s_countMoveConstruct;
        static uint64_t s_countCopyAssign;
        static uint64_t s_countMoveAssign;
        static uint64_t s_countLess;
        static uint64_t s_countHash;
        static uint64_t s_countEquals;

        Element() : value(0) { ++s_countDefaultConstruct; }
        Element(value_type val) : value(val) { }
        Element(const Element& other): value(other.value) { ++s_countCopyConstruct; }
        Element(Element&& other): value(other.value) { ++s_countMoveConstruct; }
        Element& operator=(const Element& other) {
            value = other.value;
            ++s_countCopyAssign;
            return *this;
        }
        Element& operator=(Element&& other) {
            value = other.value;
            ++s_countMoveAssign;
            return *this;
        }

        static void resetCounts() {
            s_countDefaultConstruct = 0;
            s_countCopyConstruct = 0;
            s_countMoveConstruct = 0;
            s_countCopyAssign = 0;
            s_countMoveAssign = 0;
            s_countLess = 0;
            s_countHash = 0;
            s_countEquals = 0;
        }
    };

    template<class T> uint64_t Element<T>::s_countDefaultConstruct;
    template<class T> uint64_t Element<T>::s_countCopyConstruct;
    template<class T> uint64_t Element<T>::s_countMoveConstruct;
    template<class T> uint64_t Element<T>::s_countCopyAssign;
    template<class T> uint64_t Element<T>::s_countMoveAssign;
    template<class T> uint64_t Element<T>::s_countLess;
    template<class T> uint64_t Element<T>::s_countHash;
    template<class T> uint64_t Element<T>::s_countEquals;

    template<class T>
    bool operator<(const Element<T>& left, const Element<T>& right) {
        ++Element<T>::s_countLess;
        return left.value < right.value;
    }

    template<class T>
    bool operator==(const Element<T>& left, const Element<T>& right) {
        ++Element<T>::s_countEquals;
        return left.value == right.value;
    }

} // namespace (anonymous)
namespace std {
    template <class T>
    struct hash<Element<T>> {
        size_t operator()(const Element<T>& e) const {
            ++Element<T>::s_countHash;
            return std::hash<T>{}(e.value);
        }
    };
} // namespace std
namespace {

    ////////////////////////////////////////////////////////////////////////////////
    // Driver

    template<class T>
    struct CountHashUnique {
        size_t operator()(const std::vector<T>& items) const {
            return hash_unique(items).size();
        }
    };

    template<class T>
    struct CountSortUnique {
        size_t operator()(const std::vector<T>& items) const {
            return sort_unique(items).size();
        }
    };

    template<class T>
    struct CountCustomHashUnique {
        size_t operator()(const std::vector<T>& items) const {
            return custom_hash_unique(items).size();
        }
    };

    template<class T>
    struct TestRun {
        std::string typeName;
        size_t nUnique;
        std::vector<T> items;

        template<class Rand>
        TestRun(const std::string& typeName, size_t nTotal, size_t nUnique, Rand& engine)
            : typeName(typeName), nUnique(nUnique), items(generateTestItems<T>(nTotal, nUnique, engine)) { }

        template<template<class> class Op>
        void measure(const std::string& methodName) const {
            // Wall clock timing
            const auto start = timing_clock::now();
            const auto count = Op<T>()(items);
            const auto duration = timing_clock::now() - start;

            // Operation counter
            typedef Element<T> E;
            const auto elements = std::vector<E>(items.begin(), items.end());
            E::resetCounts();
            const auto elementCount = Op<E>()(elements);

            assert(count == nUnique);
            assert(elementCount == nUnique);
            std::cout << typeName
                      << "," << items.size()
                      << "," << nUnique
                      << "," << methodName
                      << "," << std::chrono::duration_cast<std::chrono::duration<float>>(duration).count()
                      << "," << E::s_countDefaultConstruct
                      << "," << E::s_countCopyConstruct
                      << "," << E::s_countMoveConstruct
                      << "," << E::s_countCopyAssign
                      << "," << E::s_countMoveAssign
                      << "," << E::s_countLess
                      << "," << E::s_countHash
                      << "," << E::s_countEquals
                      << "\n";
        }
    };

    template<class T, class Rand>
    void measureType(const std::string& typeName, size_t logTotal, float uniqueRatio, size_t nReps, Rand& engine) {
        const auto nTotal = 1 << logTotal;
        const auto nUnique = std::max(size_t(1), static_cast<size_t>(uniqueRatio * nTotal));
        if (nUnique <= (static_cast<size_t>(std::numeric_limits<T>::max())
                        - static_cast<size_t>(std::numeric_limits<T>::min()))) {
            for (auto i = 0u; i < nReps; ++i) {
                TestRun<T> run(typeName, nTotal, nUnique, engine);
                for (auto j = 0u; j < 3; ++j) {
                    // Use this loop to change the order each time we go
                    switch ((i + j) % 3) {
                    case 0:
                        run.template measure<CountHashUnique>("hash_unique");
                        break;
                    case 1:
                        run.template measure<CountSortUnique>("sort_unique");
                        break;
                    case 2:
                        run.template measure<CountCustomHashUnique>("custom_hash_unique");
                        break;
                    default:
                        assert(false);
                    }
                }
            }
            std::cout.flush();
        }
    }

    void measureRuntime() {
        std::mt19937_64 engine;
        std::cout << "key_type,n_total,n_unique,method,time,c_default,c_copy,c_move,c_assign,c_move_assign,c_less,c_hash,c_equals\n";
        const auto nReps = 60;
        for (const auto uniqueRatio : {0.015625, .25, .5, .75, 1.}) {
            for (const auto logTotal : {6, 8, 10, 12, 14, 16, 18, 20, 22, 24}) {
                measureType<int32_t>("int32", logTotal, uniqueRatio, nReps, engine);
            }
            for (const auto logTotal : {6, 8, 10, 12, 14, 16, 18}) {
                measureType<uint8_t>("uint8", logTotal, uniqueRatio, nReps, engine);
                measureType<uint16_t>("uint16", logTotal, uniqueRatio, nReps, engine);
                measureType<uint32_t>("uint32", logTotal, uniqueRatio, nReps, engine);
                measureType<uint64_t>("uint64", logTotal, uniqueRatio, nReps, engine);
            }
        }
    }

} // namespace (anonymous)


int main() {
    measureRuntime();
    return 0;
}
