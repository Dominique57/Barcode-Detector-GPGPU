#include <stdexcept>
#include "main.hh"
#include "entryPoint.hh"

namespace po = boost::program_options;

int main(int argc, char** argv)
{
    try {
        auto const& desc = define_options();
        auto const& vm = parse_options(desc, argc, argv);
        return run(desc, vm);
    } catch (po::error &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 2;
    }
}

po::options_description define_options()
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "show usage");
    desc.add_options()
        ("database,d", po::value<std::string>()->required(), "Databse of lbp centrois\n  - input is one database path");
    desc.add_options()
        ("image,i", po::value<std::string>(), "Predict an image\n  - input is one image path");
    desc.add_options()
        ("video,v", po::value<std::string>(), "Predict a video\n  - input is one video path");
    desc.add_options()
        ("camera,c", po::value<unsigned>()->implicit_value(0), "Predict from live feed\n  - input is opencv camera identifier as unsigned");
    desc.add_options()
        ("gen-predicted-rgb,g", po::value<std::string>(), "Generates a predicted image with one color for each class");
    desc.add_options()
        ("gen-lbp,l", po::value<std::vector<std::string>>()->multitoken(), "Generates the matrix of hisotgram of the lbp algorithm\n  - input is a list of image paths\n  - output is the same path with '.txt' added at the end");
    desc.add_options()
        ("test,t", "Compare CPU and GPU implementations for safety checks");
    return desc;
}

po::variables_map parse_options(const po::options_description& desc, int argc,
        char** argv)
{
    po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

    return vm;
}

int run(const po::options_description& desc, const po::variables_map& vm)
{
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    if (vm["image"].empty() && vm["video"].empty() && vm["camera"].empty()
        && vm["test"].empty() && vm["gen-predicted-rgb"].empty() && vm["gen-lbp"].empty()) {
        std::cerr << "[ERROR] No image or video path given !" << std::endl;
        return 1;
    }

    if (!vm["camera"].empty())
        handleCamera(vm["database"].as<std::string>(), vm["camera"].as<unsigned>());
    if (!vm["image"].empty())
        handleImage(vm["database"].as<std::string>(), vm["image"].as<std::string>());
    if (!vm["video"].empty())
        handleVideo(vm["database"].as<std::string>(), vm["video"].as<std::string>());
    if (!vm["test"].empty())
        executeAlgorithm(vm["database"].as<std::string>(), "test.png");
    if (!vm["gen-predicted-rgb"].empty())
        generatePredictedRgb(vm["database"].as<std::string>(), vm["gen-predicted-rgb"].as<std::string>());
    if (!vm["gen-lbp"].empty())
        generateLbpOutFile(vm["database"].as<std::string>(), vm["gen-lbp"].as<std::vector<std::string>>());
    return 0;
}
