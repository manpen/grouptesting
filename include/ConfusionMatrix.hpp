#pragma once

#include <ostream>
#include <unordered_set>

#include <range/v3/action.hpp>
#include <range/v3/algorithm.hpp>
#include <range/v3/view.hpp>

struct ConfusionMatrix {
    size_t true_positive()  const {return true_positive_;}
    size_t false_positive() const {return false_positive_;}
    size_t true_negative()  const {return true_negative_;}
    size_t false_negative() const {return false_negative_;}

    size_t size_true() const {
        return true_positive() + true_negative();
    }

    size_t size_false() const {
        return false_positive() + false_negative();
    }

    size_t size() const {
        return size_true() + size_false();
    }

    double accuracy() const {
        return (1.0 * size_true()) / size();
    }

    bool correct() const {
        return !size_false();
    }

    template <typename T>
    static auto compute(const std::unordered_set<T>& truth, const std::unordered_set<T>& data, const size_t total) {
        using namespace ranges;
        ConfusionMatrix mat;

        for(auto x : truth) mat.true_positive_ += !!data.count(x);

        mat.false_positive_ = data.size() - mat.true_positive_;
        mat.false_negative_ = truth.size() - mat.true_positive_;
        mat.true_negative_ = total - mat.size();

        return mat;
    }

private:
    size_t true_positive_{0};
    size_t false_positive_{0};
    size_t true_negative_{0};
    size_t false_negative_{0};

};

std::ostream& operator<<(std::ostream& os, const ConfusionMatrix& mat) {
    std::stringstream ss;
    ss << '[';

    auto print = [&](std::string_view view, auto num) {
        ss << view << ": " << num << " (" << (100.0 * num / mat.size()) << "%)";
    };

    print("TP", mat.true_positive());
    ss << ", ";
    print("TN", mat.true_negative());
    ss << ", ";
    print("FP", mat.false_positive());
    ss << ", ";
    print("FN", mat.false_negative());
    ss << ", ";
    ss << "ACC: " << mat.accuracy();

    ss << ']';
    return os << ss.str();
}
