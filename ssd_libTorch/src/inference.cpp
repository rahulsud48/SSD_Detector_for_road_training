#include <torch/torch.h>
#include <torch/script.h>


#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

#include <iostream>

using namespace std;

void saveTensor(const torch::Tensor& tensor, const std::string& filename) {
    // Open a binary output file stream
    std::ofstream out(filename, std::ios::binary);

    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing.");
    }

    // Get the data pointer and total number of elements in the tensor
    auto tensor_data = tensor.data_ptr<float>(); // Assuming tensor contains float data
    auto num_elements = tensor.numel();

    // Write the data to the file
    out.write(reinterpret_cast<char*>(tensor_data), num_elements * sizeof(float));

    // Close the file stream
    out.close();
}


torch::Tensor loadTensor(const std::string& filename, const std::vector<int64_t>& shape) {
    // Open a binary input file stream
    std::ifstream in(filename, std::ios::binary);

    if (!in.is_open()) {
        throw std::runtime_error("Failed to open file for reading.");
    }

    // Calculate the number of elements based on the given shape
    auto num_elements = 1;
    for (auto s : shape) {
        num_elements *= s;
    }

    // Create a tensor with the expected shape
    torch::Tensor tensor = torch::empty(shape, torch::kFloat);

    // Get the data pointer from the tensor
    auto tensor_data = tensor.data_ptr<float>();

    // Read the data from the file into the tensor
    in.read(reinterpret_cast<char*>(tensor_data), num_elements * sizeof(float));

    // Close the file stream
    in.close();

    return tensor;
}

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


    std::vector<torch::jit::IValue> jit_input;
    jit_input.push_back(img_tensor);

    auto outputs = ssd_detector.forward(jit_input).toTuple();
    torch::Tensor boxes = outputs->elements()[0].toTensor();
    torch::Tensor classes = outputs->elements()[1].toTensor();
    std::cout<<"Output size is "<<boxes.sizes()<<std::endl;
    std::cout<<"Output size is "<<classes.sizes()<<std::endl;

    saveTensor(boxes, "boxes.bin");
    saveTensor(classes, "classes.bin");


    std::vector<int64_t> boxes_shape = {1, 17451, 4};
    std::vector<int64_t> classes_shape = {1, 17451, 12};

    torch::Tensor boxes_loaded = loadTensor("boxes.bin",boxes_shape);
    torch::Tensor classes_loaded = loadTensor("classes.bin", classes_shape);

    std::cout<<"Output size is loaded boxes"<<boxes_loaded.sizes()<<std::endl;
    std::cout<<"Output size is loaded classes"<<classes_loaded.sizes()<<std::endl;


    return 0;

}