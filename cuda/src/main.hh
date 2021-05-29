#pragma once

#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;


/// Define input
po::options_description define_options();

/// Get input
po::variables_map parse_options(const po::options_description& decs, int argc,
        char** argv);

/// Read input and execute accordingly
void run(const po::options_description& desc, const po::variables_map& vm);
