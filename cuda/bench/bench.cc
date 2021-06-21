#include <benchmark/benchmark.h>
#include <entryPoint.hh>
#include <lbp/lbpCpu.hh>
#include <lbp/lbpGpu.hh>
#include <knn/knnCpu.hh>
#include <knn/knnGpu.hh>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>

cv::Mat_<uchar> image = cv::imread("test.png", cv::IMREAD_GRAYSCALE);
auto lbpCpuImg = LbpCpu(image.cols, image.rows);
auto lbpGpuImg = LbpGpu(image.cols, image.rows);

cv::VideoCapture cap("test.mp4");
auto vidWidth = (unsigned)(cap.get(cv::CAP_PROP_FRAME_WIDTH));
auto vidHeight = (unsigned)(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
auto framecount = (unsigned )(cap.get(cv::CAP_PROP_FRAME_COUNT));
auto lbpCpuVid = LbpCpu(vidWidth, vidHeight);
auto lbpGpuVid = LbpGpu(vidWidth, vidHeight);

auto knnCpu = KnnCpu("kmeans.database", 16, 256);
auto knnGpu = KnnGpu("kmeans.database", 16, 256);

void BM_Rendering_cpu_image(benchmark::State& st) {
    auto labelsCpu = std::vector<uchar>(lbpCpuImg.numberOfPatches());
    for (auto _ : st) {
        lbpCpuImg.run(image);
        knnCpu.transform(lbpCpuImg.getFeatures(), labelsCpu);
    }
    st.counters["frame_rate"] = benchmark::Counter((double)st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_cpu_image_withInit(benchmark::State& st) {
    for (auto _ : st) {
        auto lbpCpuImg = LbpCpu(image.cols, image.rows);
        auto knnCpu = KnnCpu("kmeans.database", 16, 256);
        auto labelsCpu = std::vector<uchar>(lbpCpuImg.numberOfPatches());

        lbpCpuImg.run(image);
        knnCpu.transform(lbpCpuImg.getFeatures(), labelsCpu);
    }
    st.counters["frame_rate"] = benchmark::Counter((double)st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu_image(benchmark::State& st) {
    auto labelsGpu = std::vector<uchar>(lbpCpuImg.numberOfPatches());
    for (auto _ : st) {
        lbpGpuImg.run(image);
        knnGpu.transformMultiStep(lbpGpuImg.getCudaFeatures(), labelsGpu);
    }
    st.counters["frame_rate"] = benchmark::Counter((double)st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu_image_withInit(benchmark::State& st) {
    for (auto _ : st) {
        auto lbpGpuVid = LbpGpu(vidWidth, vidHeight);
        auto knnGpu = KnnGpu("kmeans.database", 16, 256);
        auto labelsGpu = std::vector<uchar>(lbpCpuImg.numberOfPatches());

        lbpGpuImg.run(image);
        knnGpu.transformMultiStep(lbpGpuImg.getCudaFeatures(), labelsGpu);
    }
    st.counters["frame_rate"] = benchmark::Counter((double)st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu_video(benchmark::State& st) {
    auto labelsGpu = std::vector<uchar>(lbpGpuImg.numberOfPatches());
    cv::Mat frame;
    cv::Mat_<uchar> Gray_frame;

    for (auto _ : st) {
        if (!cap.read(frame)) {
            break;
        }
        cv::cvtColor(frame, Gray_frame, cv::COLOR_BGR2GRAY);
        lbpGpuVid.run(Gray_frame);
        knnGpu.transformMultiStep(lbpGpuVid.getCudaFeatures(), labelsGpu);
    }

    st.counters["frame_rate"] = benchmark::Counter((double)st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Rendering_cpu_image)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Rendering_cpu_image_withInit)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Rendering_gpu_image)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Rendering_gpu_image_withInit)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_Rendering_gpu_video)
    ->Iterations(framecount)
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();


BENCHMARK_MAIN();