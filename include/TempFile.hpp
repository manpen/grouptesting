#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <unistd.h>

std::string makeRandomFile(const std::string &prefix) {
    auto filename = std::make_unique<char[]>(prefix.size() + 8);
    std::copy_n(prefix.cbegin(), prefix.size(), filename.get());
    std::fill_n(filename.get() + prefix.size(), 6, 'X');
    filename[prefix.size() + 7] = 0;

    int fh = mkstemp(filename.get());
    if (fh == -1)
        return "";
    close(fh);

    return {filename.get()};
}
