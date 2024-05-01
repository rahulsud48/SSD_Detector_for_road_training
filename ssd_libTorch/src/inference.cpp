#include <torch/torch.h>
#include <torch/script.h>


#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <cmath> 

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

// data encoder functions

class DataEncoder
{
private:
    int img_w; 
    int img_h;
    int num_classes;
    int num_fms = 5;
    float cls_threshold=0.7f;
    float nms_threshold=0.3f;
    std::vector<float> anchor_areas;
    std::vector<float> aspect_ratios{0.5,1,2};
    std::vector<float> scales;
    std::vector<int> fm_sizes;
    torch::Tensor anchor_boxes_tensor;
    // std::vector<torch::Tensor> anchors;
    // std::vector<torch::Tensor> anchor_grid;
    // std::vector<torch::Tensor> anchor_boxes;
    void createParameters()
    {
        // creating areas for anchor boxes
        for (int i=3; i<8; i++)
        {
            anchor_areas.push_back(std::pow(std::pow(2,i),2));
        }
        // creating scales
        for (int i=0; i<3; i++)
        {
            float power = (float)((float)i/3);
            scales.push_back(std::pow(2.,(float)((float)i/3)));
        }
        // creating feature maps sizes
        for (int i=0; i<5; i++)
        {
            fm_sizes.push_back(std::ceil(300/std::pow(2.0, i+3)));
        }
        int i = 0;
        std::vector<torch::Tensor> anchor_boxes;
        for (const auto& fm_size : fm_sizes)
        {
            torch::Tensor anchors = generate_anchors(anchor_areas[i], aspect_ratios, scales);
            torch::Tensor anchor_grid = generate_anchor_grid(img_w, img_h, fm_size, anchors);
            anchor_boxes.push_back(anchor_grid);
            i++;
        }
        anchor_boxes_tensor = torch::cat(anchor_boxes, 0);
        std::cout << "Shape of concatenated tensor: " << anchor_boxes_tensor.sizes() << std::endl;
    }

    torch::Tensor generate_anchor_grid(int img_w, int img_h, int fm_size, torch::Tensor anchors)
    {
        float grid_size = (float)img_w/(float)fm_size;
        std::vector<torch::Tensor> meshgrid = torch::meshgrid({torch::arange(0, fm_size) * grid_size, torch::arange(0, fm_size) * grid_size});
        
        anchors = anchors.view({-1, 1, 1, 4});
        torch::Tensor xyxy = torch::stack({meshgrid[0], meshgrid[1], meshgrid[0], meshgrid[1]}, 2).to(torch::kFloat);
        auto boxes = (xyxy + anchors).permute({2, 1, 0, 3}).contiguous().view({-1, 4});
        // Clamp the coordinates to the input size
        boxes.index({torch::indexing::Slice(), torch::indexing::Slice(0, -1, 2)}).clamp_(0, img_w);
        boxes.index({torch::indexing::Slice(), torch::indexing::Slice(1, -1, 2)}).clamp_(0, img_h);
        return boxes;

    }

    torch::Tensor generate_anchors(float anchor_area, std::vector<float> aspect_ratios, std::vector<float> scales)
    {
        std::vector<std::vector<float>> anchors;
        for (const auto& scale : scales )
        {
            for (const auto& ratio : aspect_ratios)
            {
                float h = std::round(std::pow(anchor_area,0.5)/ratio);
                float w = std::round(ratio*h);
                float x1 = (std::pow(anchor_area,0.5) - scale * w) * 0.5f;
                float x2 = (std::pow(anchor_area,0.5) + scale * w) * 0.5f;
                float y1 = (std::pow(anchor_area,0.5) - scale * h) * 0.5f;
                float y2 = (std::pow(anchor_area,0.5) + scale * h) * 0.5f;
                anchors.push_back({x1,y1,x2,y2});
            }
        }
        torch::Tensor anchors_tensor = convert_2dvec_to_torch_tensor(anchors);
        return anchors_tensor;
    }

    torch::Tensor convert_2dvec_to_torch_tensor(std::vector<std::vector<float>> array_2d)
    {
        // use template to handle other data structures
        std::vector<float> flat_array;
        for (const auto& inner_vec : array_2d) 
        {
            flat_array.insert(flat_array.end(), inner_vec.begin(), inner_vec.end());
        }
        auto tensor = torch::tensor(flat_array, torch::kFloat);
        // Reshape the tensor to match the original 2D structure
        int rows = array_2d.size();  // Number of outer vectors
        int cols = array_2d[0].size();  // Assumes all inner vectors have the same size
        torch::Tensor reshaped_tensor = tensor.view({rows, cols});
        return reshaped_tensor;
    }

public:
    DataEncoder(std::map<std::string,int> img_size, int num_classes)
    {
        this->img_w = img_size["width"];
        this->img_h = img_size["height"];
        this->num_classes = num_classes;
        createParameters();
    }

    torch::Tensor decode_boxes(const torch::Tensor& deltas, const torch::Tensor& anchors) {
        // Calculate the width and height of the anchors
        auto anchors_wh = anchors.slice(1, 2, 4) - anchors.slice(1, 0, 2) + 1;

        // Calculate the centers of the anchors
        auto anchors_ctr = anchors.slice(1, 0, 2) + 0.5 * anchors_wh;

        // Calculate the centers of the predicted boxes
        auto pred_ctr = deltas.slice(1, 0, 2) * anchors_wh + anchors_ctr;

        // Calculate the width and height of the predicted boxes
        auto pred_wh = torch::exp(deltas.slice(1, 2, 4)) * anchors_wh;

        // Calculate the top-left and bottom-right coordinates
        auto top_left = pred_ctr - 0.5 * pred_wh;
        auto bottom_right = pred_ctr + 0.5 * pred_wh - 1;

        // Concatenate the top-left and bottom-right coordinates
        auto result = torch::cat({top_left, bottom_right}, 1);

        return result;
    }

    void decode(torch::Tensor loc_pred, torch::Tensor cls_pred, int batch_size)
    {
        for (int i=0; i<batch_size; i++)
        {
            torch::Tensor boxes = decode_boxes(loc_pred[i], anchor_boxes_tensor);
            torch::Tensor conf = cls_pred[i].softmax(1);
            for (int j=1; j<num_classes;j++)
            {
                torch::Tensor class_conf = conf.index({torch::indexing::Slice(), 1});
                // Find indices where class_conf exceeds cls_threshold
                auto ids_tensor = (class_conf > cls_threshold).nonzero();

                // Squeeze the tensor to remove extra dimensions
                auto ids_squeezed = ids_tensor.squeeze();

                // Convert to a list of indices
                std::vector<int64_t> ids_list;
                ids_squeezed = ids_squeezed.view(-1);  // Ensure tensor is 1D

                ids_list.assign(ids_squeezed.data_ptr<int64_t>(), ids_squeezed.data_ptr<int64_t>() + ids_squeezed.numel());
            }
            std::cout<<boxes.sizes()<<"\n";
        }
        
    }
};



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

    // saveTensor(boxes, "boxes.bin");
    // saveTensor(classes, "classes.bin");


    // std::vector<int64_t> boxes_shape = {1, 17451, 4};
    // std::vector<int64_t> classes_shape = {1, 17451, 12};

    // torch::Tensor boxes_loaded = loadTensor("boxes.bin",boxes_shape);
    // torch::Tensor classes_loaded = loadTensor("classes.bin", classes_shape);

    // std::cout<<"Output size is loaded boxes"<<boxes_loaded.sizes()<<std::endl;
    // std::cout<<"Output size is loaded classes"<<classes_loaded.sizes()<<std::endl;

    // vector<double> test = generate_anchor_boxes();

    // variables for data encoder class

    std::map<std::string, int> img_size;
    img_size["width"] = 300;
    img_size["height"] = 300;

    int num_classes = 12;
    int batch_size = 1;

    // std::cout<<boxes[0].sizes()<<"\n";
    // std::cout<<boxes[0]<<"\n";

    DataEncoder encoder(img_size, num_classes);
    encoder.decode(boxes, classes, batch_size);

    return 0;

}