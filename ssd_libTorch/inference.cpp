#include <torch/torch.h>
#include <torch/script.h>
// #include <torch/trace.h>
#include <iostream>

using namespace std;

int main() {

    // Load the model insde the `ssd_detector`
    // std::str model_path = "/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/Projects/autonomous_cars/Object_Detector_for_road/SSD_Detector_for_road_training/checkpoints/Detector_best.pth"
    torch::jit::script::Module ssd_detector = torch::jit::load("tiny_model.pt");

    torch::Tensor input = torch::randn({1,3,300,300});


    // Initialize the device to CPU
    torch::DeviceType device = torch::kCPU;
    // If CUDA is available,run on GPU
    if (torch::cuda::is_available())
        device = torch::kCUDA;
    cout << "Running on: "
            << (device == torch::kCUDA ? "GPU" : "CPU") << endl;

    std::vector<torch::jit::IValue> jit_input;
    jit_input.push_back(input);

    auto outputs = ssd_detector.forward(jit_input).toTuple();
    torch::Tensor out1 = outputs->elements()[0].toTensor();
    torch::Tensor out2 = outputs->elements()[1].toTensor();
    std::cout<<"Output size is "<<out1.sizes()<<std::endl;
    std::cout<<"Output size is "<<out2.sizes()<<std::endl;

    return 0;

}