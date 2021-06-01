#include <stdexcept>
#include <lbp/lbp.hh>
#include <kmeans/KmeansTransform.hh>
#include "main.hh"
#include "image/matrix.hh"
#include "benchmark/Bench.hh"


namespace po = boost::program_options;

int main(int argc, char** argv)
{
    auto test1 = "test CPU";
    bench::Bench::start(test1);
    try {
        auto const& desc = define_options();
        auto const& vm = parse_options(desc, argc, argv);
        run(desc, vm);
    } catch (po::error &e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 2;
    }
    bench::Bench::end(test1);
    bench::Bench::print(std::cerr, test1, "milliseconds");
    return 0;
}

po::options_description define_options()
{
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "show usage");
    desc.add_options()
        ("image,i", po::value<std::string>(), "input image path");
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

void run(const po::options_description& desc, const po::variables_map& vm)
{
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return;
    } else if (vm["image"].empty()) {
        std::cerr << "No image path given" << std::endl;
        return;
    }

    auto image_path = vm["image"].as<std::string>().c_str();
    auto image = Matrix<>(4032, 3024);
    Matrix<>::readMatrix(image_path, image);

    // Execute lbp
    auto lbp = Lbp(image);
    auto features = lbp.run();

    // Execute kmeans
    auto labels = Matrix<>(1, lbp.patchCount());
    KmeansTransform kmeans("kmeans.database", 16, lbp.textonSize());
    kmeans.transform(features, labels);

    // Extract barcode position
}