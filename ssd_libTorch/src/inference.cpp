#include <torch/torch.h>
#include <torch/script.h>


#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

#include <iostream>

using namespace std;

void load_model(std::string* model_path, torch::jit::script::Module& ssd_detector)
{
    ssd_detector = torch::jit::load(*model_path);
}

void load_image(std::string* image_path, cv::Mat* img)
{
    *img = cv::imread(*image_path, cv::IMREAD_COLOR);
}

void transform_image(cv::Mat* img, torch::Tensor* img_tensor)
{
    // Resize to a new fixed size (e.g., 100x100 pixels)
    cv::Size new_size(300, 300);
    cv::resize(*img, *img, new_size);

    // Convert cv::Mat to a PyTorch tensor
    *img_tensor = torch::from_blob(
        img->data, {img->rows, img->cols, img->channels()}, torch::kByte);


    // Convert to float and scale pixel values from [0, 255] to [0.0, 1.0]
    *img_tensor = img_tensor->to(torch::kFloat).div(255.0);

    // Convert from BGR to RGB by reversing the last dimension
    *img_tensor = img_tensor->permute({2, 0, 1});  // [channels, height, width]

    // Add a batch dimension (batch size = 1)
    *img_tensor = img_tensor->unsqueeze(0);  // [1, channels, height, width]
}




int main() {

    // Load the model insde the `ssd_detector`
    std::string model_path = "/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/Projects/autonomous_cars/Object_Detector_for_road/SSD_Detector_for_road_training/ssd_libTorch/build/tiny_model.pt";
    // load test image
    std::string image_path = "/media/rahul/a079ceb2-fd12-43c5-b844-a832f31d5a39/Projects/autonomous_cars/Object_Detector_for_road/SSD_Detector_for_road_training/ssd_libTorch/build/test_image.jpg";

    torch::jit::script::Module ssd_detector;
    load_model(&model_path, ssd_detector);

    cv::Mat img;
    load_image(&image_path, &img);

    torch::Tensor img_tensor;
    transform_image(&img, &img_tensor);


    // cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    // Resize to a new fixed size (e.g., 100x100 pixels)
    // cv::Size new_size(300, 300);
    // cv::resize(img, img, new_size);

    // // Convert cv::Mat to a PyTorch tensor
    // torch::Tensor img_tensor = torch::from_blob(
    //     img.data, {img.rows, img.cols, img.channels()}, torch::kByte);


    // // Convert to float and scale pixel values from [0, 255] to [0.0, 1.0]
    // img_tensor = img_tensor.to(torch::kFloat).div(255.0);

    // // Convert from BGR to RGB by reversing the last dimension
    // img_tensor = img_tensor.permute({2, 0, 1});  // [channels, height, width]

    // // Add a batch dimension (batch size = 1)
    // img_tensor = img_tensor.unsqueeze(0);  // [1, channels, height, width]



    // torch::Tensor input = torch::randn({64,3,300,300});


    // // Initialize the device to CPU
    // torch::DeviceType device = torch::kCPU;
    // // If CUDA is available,run on GPU
    // if (torch::cuda::is_available())
    //     device = torch::kCUDA;
    // cout << "Running on: "
    //         << (device == torch::kCUDA ? "GPU" : "CPU") << endl;

    std::vector<torch::jit::IValue> jit_input;
    jit_input.push_back(img_tensor);

    auto outputs = ssd_detector.forward(jit_input).toTuple();
    torch::Tensor out1 = outputs->elements()[0].toTensor();
    torch::Tensor out2 = outputs->elements()[1].toTensor();
    std::cout<<"Output size is "<<out1.sizes()<<std::endl;
    std::cout<<"Output size is "<<out2.sizes()<<std::endl;

    return 0;

}