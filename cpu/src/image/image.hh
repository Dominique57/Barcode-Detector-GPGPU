#pragma once

#include <string>
#include <stdexcept>

class Image {
public:
    Image(const std::string&);

private:
    unsigned width_;
    unsigned heigth_;
    unsigned char *data_;
};