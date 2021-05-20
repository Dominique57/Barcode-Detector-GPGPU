#pragma once

#include <string>
#include <stdexcept>

class Image {
public:
    Image(const std::string& filePath);

private:
    unsigned width_;
    unsigned heigth_;
    unsigned char *data_;
};