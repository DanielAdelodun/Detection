// Include Libraries.
#include <opencv2/opencv.hpp>
#include <fstream>
#include <cstdio>
#include <thread>
#include <cstdlib>
#include <getopt.h>
#include <string>
#include <mutex>
#include <chrono>


// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float IMAGE_WIDTH = 1280.0;
const float IMAGE_HEIGHT = 1280.0;
const float SCALE_FACTOR = 1.0 / 255.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Frame Buffer.
mutex frame_mutex;
const int FRAME_SIZE = (int)IMAGE_WIDTH * (int)IMAGE_HEIGHT * 3;
char frame_buffer[FRAME_SIZE];
int frame_no = 0;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);
Scalar WHITE = Scalar(255,255,255);

inline void parse_args(int argc, char** argv, Net& net, string& host, FILE *&pipein, vector<string>& class_list)
{
  const char* usage = "Run Yolov6 3.0. Connects to [hostname]:5001, and expects a 25fps 1280x1280 H264 encoded video stream.\n\n"
  "Usage: %s [options]\n\n"
    "Options:\n"
    "    -m, --model <path>       Path to ONNX model.\n"
    "    -c, --classname-file <path>  Path to class name file.\n"
    "    -g, --use-gpu <bool>     Use GPU or not. (default: true  [1])\n"
    "    -H, --host <address>     Host address. (default: gabriel.local)\n"
    "    -h, --help               Print this message.\n";

  ifstream ifs;
  string classfile_line;
  string model_path;
  bool use_gpu = true;

  int opt;

  while (1)
    {
      static struct option long_options[] =
        {
          {"model",           required_argument, 0, 'm'},
          {"classname-file",  required_argument, 0, 'c'},
          {"use-gpu",         required_argument, 0, 'g'},
          {"host",            required_argument, 0, 'H'},
          {"help",            no_argument,       0, 'h'},
          {0, 0, 0, 0}
        };
      opt = getopt_long (argc, argv, ":m:H:c:g:h",
                        long_options, 0);

      if (opt == -1)
        break;

      switch (opt)
        {
        case 'm':
          model_path = optarg;
          break;
        
        case 'H':
          host = optarg;
          break;

        case 'c': case 'n':
          class_list.clear();
          ifs.open(optarg);
          while (getline(ifs, classfile_line))
          {
              class_list.push_back(classfile_line);
          }
          break;

        case 'g':
          use_gpu = atoi(optarg);
          break;

        case 'h':
          printf(usage, argv[0]);
          exit(0);
          
        case '?':
          perror("Unknown option...");
          exit(1);

        case ':':
          perror("Missing option arguments...");
          exit(1);

        default:
          abort ();
        }
    }

  printf("Connecting to %s...\n", host.c_str());

  const char* ffmpeg_cmd_format = 
    "ffmpeg -nostdin -probesize 32 -flags low_delay -fflags nobuffer "
    "-codec:v %s -r 25 -i tcp://%s:5001 "
    "-pix_fmt rgb24 -an -vcodec rawvideo -f rawvideo pipe: 2>/dev/null";

  const char* codec = use_gpu ? "h264_cuvid" : "h264";

  int size_s = std::snprintf( nullptr, 0, ffmpeg_cmd_format, codec, host.c_str() ) + 1;
  if( size_s <= 0 ) {
    perror( "Error during formatting.");
    exit(1);
  }
  auto size = static_cast<size_t>( size_s );
  char ffmpeg_cmd[ size ];
  std::snprintf( ffmpeg_cmd, size, ffmpeg_cmd_format, codec, host.c_str() );

  pipein = popen(ffmpeg_cmd, "r");
  if (pipein == NULL)
  {
      perror("Error opening video stream!\n");
      exit(1);
  }

  net = readNetFromONNX(model_path.empty() ? "yolov6.onnx" : model_path);
  if (net.empty())
  {
      perror("Error loading model!\n");
      exit(1);
  }

  if (use_gpu)
  {
      net.setPreferableBackend(DNN_BACKEND_CUDA);
      net.setPreferableTarget(DNN_TARGET_CUDA);
  }
  else
  {
      net.setPreferableBackend(DNN_BACKEND_DEFAULT);
      net.setPreferableTarget(DNN_TARGET_CPU);
  }

  // Print model information.
  vector<String> outNames = net.getUnconnectedOutLayersNames();
  vector<int> outLayers = net.getUnconnectedOutLayers();
  for (int i = 0; i < outNames.size(); i++)
  {
      printf("Output Layer [%d]: %s\n", outLayers[i], outNames[i].c_str());

      MatShape netInputShape{1, 3, 640, 640};
      vector<MatShape> inLayerShape, outLayerShape;
      net.getLayerShapes(netInputShape, outLayers[i], inLayerShape, outLayerShape);

      printf("Output Layer Shape: ");
      for (int j = 0; j < outLayerShape.size(); j++) {
        for (int k = 0; k < outLayerShape[j].size(); k++)
          printf("%d (%d, %d), ", outLayerShape[j][k], j, k);
      }
      printf("\n");
  }
  // Output layer size.

  printf("Connected and loaded model!\n");
}

inline size_t read_frame(FILE * pipein)
{
    frame_mutex.lock();
    size_t ret = fread(frame_buffer, 1, FRAME_SIZE, pipein);
    frame_mutex.unlock();
    frame_no++;
    return ret;
}

void read_frames(FILE *pipein)
{
    for(;;)
    {
        if (read_frame(pipein) < FRAME_SIZE)
        {
            printf("Error receiving frame!\n");
            break;
        }
        this_thread::sleep_for(chrono::milliseconds(1));
    }
}


void draw_label(Mat& input_image, string label, int left, int top)
{
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height); // Ensures that the label is within the image.
    Point tlc = Point(left, top);
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, WHITE, THICKNESS);
}


vector<Mat> pre_process(Mat &input_image, Net &net)
{
    Mat blob;
    blobFromImage(input_image, blob, SCALE_FACTOR, Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}


Mat post_process_yolo(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<float> cls_scores;
    vector<Rect> boxes;

    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    

    float *data = (float *)outputs[0].data;

    const int dimensions = 85;
    const int rows = 8400;
    // Iterate through 8400 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire index of best class score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
            {
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                cls_scores.push_back(max_class_score);

                // Center.
                float cx = data[0];
                float cy = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Scaled Bounding box coordinates.
                int left = int((cx - w/2) * x_factor);
                int top = int((cy - h/2) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                
                boxes.push_back(Rect(left, top, width, height));
            }

        }
        // Jump to the next column.
        data += 85;
    }

    // Perform Non Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, cls_scores, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    // Print class name and confidence.
    // Print bounding box coordinates.
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        printf("%s: %.2f\n", class_name[class_ids[idx]].c_str(), cls_scores[idx]);
        printf("Bounding Box: (%d, %d) (%d, %d)\n", box.x, box.y, box.width, box.height);
    }

    // Draw detections.
    for (int i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];

        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", cls_scores[idx]);
        label = class_name[class_ids[idx]] + ": " + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}

int main(int argc, char** argv)
{

    // Initialize variables.
    Net net;
    string host = "gabriel.local";
    FILE *pipein = NULL;
    vector<string> class_list = {
        "person", "bicycle", "car", "motorbike", "aeroplane", 
        "bus", "train", "truck", "boat", "traffic light", "fire hydrant", 
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", 
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", 
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", 
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", 
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", 
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", 
        "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", 
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", 
        "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };

    parse_args(argc, argv, net, host, pipein, class_list);

    // Wait for the first frame.
    size_t bytes = read_frame(pipein);
    if (bytes < FRAME_SIZE)
    {
        printf("Error receiving first frame!\n");
        return -1;
    }
    printf("Received first frame!\n");
    thread t1(read_frames, pipein);

    // Opencv Image Buffer.
    Mat frame(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, frame_buffer);
    
    double freq = getTickFrequency() / 1000;
    Mat img;
    for(;;)
    {
        frame_mutex.lock();
        Mat input = frame.clone();
        frame_mutex.unlock();

        vector<Mat> detections;
        detections = pre_process(input, net);
        img = post_process_yolo(input, detections, class_list).clone();

        vector<double> layersTimes;
        double t = net.getPerfProfile(layersTimes);
        cout << format("Frame [%d]:\t%.2f\tms", frame_no + 1, t / freq) << endl;

        // Convert to BGR.
        cvtColor(img, img, COLOR_RGB2BGR);
        imshow("frame", img);
        if ((char)waitKey(1) == 'q')
            break;
    }

    // Release resources.
    destroyAllWindows();
    terminate();

    return 0;
}
