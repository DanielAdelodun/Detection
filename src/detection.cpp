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
#include <signal.h>
#include "servo.hpp"

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

const char *usage = "Run Yolov6 3.0 or Facebook's DETR.\nConnect to [host]:5001 and read a H264 encoded video stream.\n\n"
                      "Usage: %s [options]\n\n"
                      "Options:\n"
                      "    -m, --model <path>                             Path to ONNX model. (default: ../models/model.onnx\n"
                      "    -c, --classname-file <path>                    Path to class name file (default: ../COCO Classnames/coco.names).\n"
                      "    -g, --use-gpu <bool>                           Use GPU or not. (default: true  [1])\n"
                      "    -a, --host|address <address>                   Host address. (default: gabriel.local)\n"
                      "    -s, --original-image-size <width>x<height>     Original image size. (default: 1280x1280)\n"
                      "    -S, --input-image-size <width>x<height>        ONNX model's input image size. (default: 640x640)\n"
                      "    -t, --score-threshold <float>                  Score threshold. (default: 0.5)\n"
                      "    -k, --mean <float>,<float>,<float>             Mean values. Comma seperated, no whitespace. (default: 0,0,0)\n"
                      "    -d, --std  <float>,<float>,<float>             Standard deviation. (default: 1,1,1)\n"
                      "    -y  --yolov6 <int>                             [Yolov6] detection output format. (default: true [1])\n"
                      "    -h, --help                                     Print this message.\n";

class Args
{
  public:
  Args (int argc, char **argv):
  model_path("../models/yolov6.onnx"),
  class_name_file("../COCO Classnames/coco.names"),
  use_gpu(true),
  host("gabriel.local"),
  port("5001"),
  original_image_width(1280.0),
  original_image_height(1280.0),
  input_width(640.0),
  input_height(640.0),
  score_threshold(0.5),
  mean(Scalar (0.0, 0.0, 0.0)),
  standard_deviation(Scalar (1.0, 1.0, 1.0)),
  yolov6(1)
  {
    static struct option long_options[] =
    {
      {"model", required_argument, 0, 'm'},
      {"classname-file", required_argument, 0, 'c'},
      {"use-gpu", required_argument, 0, 'g'},
      {"host", required_argument, 0, 'a'},
      {"address", required_argument, 0, 'a'},
      {"port", required_argument, 0, 'p'},
      {"original-image-size", required_argument, 0, 's'},
      {"input-image-size", required_argument, 0, 'S'},
      {"score-threshold", required_argument, 0, 't'},
      {"mean", required_argument, 0, 'k'},
      {"std", required_argument, 0, 'd'},
      {"standard-deviation", required_argument, 0, 'd'},
      {"yolov6", required_argument, 0, 'y'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}
    };

    int opt;

    while (1)
    {
      opt = getopt_long(argc, argv, "m:c:g:a:p:s:S:t:k:d:y:h",
                      long_options, 0);
      if (opt == -1)
        break;

      switch (opt)
      {
      case 'm':
        model_path = optarg;
        break;

      case 'a':
        host = optarg;
        break;

      case 'p':
        port = optarg;
        break;

      case 'c': case 'n':
        class_name_file = optarg;
        break;

      case 'g':
        use_gpu = atoi(optarg);
        break;

      case 's':
        if (sscanf(optarg, "%fx%f", &original_image_width, &original_image_height) != 2)
        {
          fprintf(stderr, "Invalid image size!");
          fprintf(stderr, usage, argv[0]);
          exit(1);
        }
        break;

      case 'S':
        if (sscanf(optarg, "%fx%f", &input_width, &input_height) != 2)
        {
          fprintf(stderr, "Invalid model input size!");
          fprintf(stderr, usage, argv[0]);
          exit(1);
        }
        break;

      case 't':
        if (sscanf(optarg, "%f", &score_threshold) != 1)
        {
          fprintf(stderr, "Invalid score threshold!");
          fprintf(stderr, usage, argv[0]);
          exit(1);
        }
        break;

      case 'k':
        if (sscanf(optarg, "%lf,%lf,%lf", &mean[0], &mean[1], &mean[2]) != 3)
        {
          fprintf(stderr, "Invalid mean value!");
          fprintf(stderr, usage, argv[0]);
          exit(1);
        }
        break;

      case 'd':
        if (sscanf(optarg, "%lf,%lf,%lf", &standard_deviation[0], &standard_deviation[1], &standard_deviation[2]) != 3)
        {
          fprintf(stderr, "Invalid standard deviation!\n");
          fprintf(stderr, usage, argv[0]);
          exit(1);
        }
        break;
        
      case 'y':
        yolov6 = atoi(optarg);
        break;

      case 'h':
        printf(usage, argv[0]);
        exit(0);

      case '?':
        fprintf(stderr, "Unknown option!\n ");
        fprintf(stderr, usage, argv[0]);
        exit(1);

      case ':':
        fprintf(stderr, "Missing argument!\n");
        fprintf(stderr, usage, argv[0]);
        exit(1);
      }
    }
  }

  string model_path;
  string class_name_file;
  bool use_gpu;
  string host;
  string port;
  float original_image_width;
  float original_image_height;
  float input_width;
  float input_height;
  float score_threshold;
  Scalar mean;
  Scalar standard_deviation;
  int yolov6;

};

const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

vector<string> class_list;


// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0, 0, 0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0, 0, 255);
Scalar WHITE = Scalar(255, 255, 255);

mutex frame_mutex;
int FRAME_SIZE;
int frame_no = 0;

class Detection
{
public:
  Detection() {}
  Detection(int class_id, float score, Rect box)
  {
    this->class_id = class_id;
    this->score = score;
    this->box = box;
  }

  int class_id;
  float score;
  Rect box;
};

Mat softmax(Mat &input, float max_value)
{
  Mat output;
  output = input - max_value;
  exp(input, output);
  float matsum = sum(output)[0];
  output = output / matsum;
  return output;
}

// Exit Handlers.

void stop(int exit_code, void * arg)
{
  printf("\ntidying up\n");
  
  Gimbal *gimbal_ptr = (Gimbal *)arg;
  gimbal_ptr->disconnect();
  
  _exit(0);
}

void cntl_c(int sig)
{
  exit(0);   
}

inline FILE* connect_to_video_host(string host, string port, bool use_gpu)
{
    const char* ffmpeg_cmd_format = 
      "ffmpeg -nostdin -probesize 32 -flags low_delay -fflags nobuffer "
      "-codec:v %s -r 25 -i tcp://%s:%s "
      "-pix_fmt rgb24 -an -vcodec rawvideo -f rawvideo pipe: 2>/dev/null";

    const char* codec = use_gpu ? "h264_cuvid" : "h264";

    int size_s = std::snprintf( nullptr, 0, ffmpeg_cmd_format, codec, host.c_str(), port.c_str() ) + 1;
    if( size_s <= 0 ) {
      fprintf(stderr, "Error formating ffmpeg command\n" );
      exit(1);
    }
    auto size = static_cast<size_t>( size_s );
    char ffmpeg_cmd[ size ];
    std::snprintf( ffmpeg_cmd, size, ffmpeg_cmd_format, codec, host.c_str(), port.c_str() );

    FILE* video_stream = popen(ffmpeg_cmd, "r");
    if (video_stream == NULL)
    {
        fprintf(stderr, "Error starting ffmpeg\n");
        exit(1);
  }

  printf("Connected to video stream at: %s\n", host.c_str());

  return video_stream;
}

inline size_t read_frame(FILE *pipein, char *frame_buffer)
{
  frame_mutex.lock();
  size_t ret = fread(frame_buffer, 1, FRAME_SIZE, pipein);
  frame_mutex.unlock();
  frame_no++;
  return ret;
}

void read_frames(FILE *pipein, char *frame_buffer)
{
  for (;;)
  {
    if (read_frame(pipein, frame_buffer) < FRAME_SIZE)
    {
      printf("Error receiving frame!\n");
      break;
    }
    this_thread::sleep_for(chrono::milliseconds(1));
  }
}

void draw_label(Mat &image, string label, int left, int top)
{
  int baseLine;
  Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
  top = max(top, label_size.height); // Ensures that the label is within the image.
  Point tlc = Point(left, top);
  Point brc = Point(left + label_size.width, top + label_size.height + baseLine);

  rectangle(image, tlc, brc, BLACK, FILLED);
  putText(image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, WHITE, THICKNESS);
}

Mat draw_detections(Mat &image, vector<Detection> &detections, const vector<string> &class_name)
{
  for (int i = 0; i < detections.size(); i++)
  {
    Detection detection = detections[i];
    int class_id = detection.class_id;
    float score = detection.score;
    Rect box = detection.box;

    // Draw bounding box.
    rectangle(image, box, BLUE, 2);

    // Draw label.
    string label = format("%s: %.2f", class_name[class_id].c_str(), score);
    draw_label(image, label, box.x, box.y);
  }
  return image;
}

vector<Mat> pre_process(InputArray image, Net &net, Scalar mean, Scalar standard_deviation, int input_width, int input_height)
{
  Mat blob;

  float first_pixel = image.getMat().ptr<float>(0, 0)[0];

  blobFromImage(image, blob, 1./255., Size(input_width, input_height), mean * 255.0, true, false);

  float first_pixel_before = blob.ptr<float>(0, 0)[0];

  for (int i = 0; i < blob.size[0]; ++i){
    float *data = blob.ptr<float>(0, i);
    for (int j = 0; j < blob.size[2] * blob.size[3]; ++j){
      data[j] = data[j] / standard_deviation[j % 3];
    }
  }

  float first_pixel_after = blob.ptr<float>(0, 0)[0];

  net.setInput(blob);

  vector<Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  return outputs;
}

vector<Detection> post_process_yolo(vector<Mat> &outputs, const vector<string> &class_name, Net &net, float score_threshold,
  int input_width, int input_height, int original_image_width, int original_image_height)
{
  // Initialize vectors to hold respective outputs while unwrapping detections.
  vector<int> class_ids;
  vector<float> confidences;
  vector<float> cls_scores;
  vector<Rect> boxes;

  // Resizing factor.
  float x_factor = original_image_width / input_width;
  float y_factor = original_image_height / input_height;

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
      float *classes_scores = data + 5;
      // Create a 1x85 Mat and store class scores of 80 classes.
      Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
      // Perform minMaxLoc and acquire index of best class score.
      Point class_id;
      double max_class_score;
      minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
      // Continue if the class score is above the threshold.
      if (max_class_score > score_threshold)
      {
        float cx = data[0];
        float cy = data[1];
        float w = data[2];
        float h = data[3];
        // Scaled Bounding box coordinates.
        int left = int((cx - w / 2) * x_factor);
        int top = int((cy - h / 2) * y_factor);
        int width = int(w * x_factor);
        int height = int(h * y_factor);

        confidences.push_back(confidence);
        class_ids.push_back(class_id.x);
        cls_scores.push_back(max_class_score);
        boxes.emplace_back(left, top, width, height);
      }
    }
    // Jump to the next column.
    data += 85;
  }

  // Perform Non Maximum Suppression and draw predictions.
  vector<int> indices;
  NMSBoxes(boxes, cls_scores, score_threshold, NMS_THRESHOLD, indices);

  vector<Detection> Detections;
  for (int i = 0; i < indices.size(); i++)
  {
    int idx = indices[i];
    Detections.emplace_back(class_ids[idx], cls_scores[idx], boxes[idx]);
  }

  return Detections;
}

vector<Detection> post_process_detr(vector<Mat> &outputs, const vector<string> &class_name, Net &net, float score_threshold,
  int input_width, int input_height, int original_image_width, int original_image_height)
{
  vector<Rect> boxes;
  vector<float> confidences;
  vector<int> indices;
  vector<int> class_ids;

  vector<Detection> detections;

  float x_factor = original_image_width;
  float y_factor = original_image_height;

  float *logits;
  float *pred_boxes;

  if (outputs.size() != 2)
  {
    perror("Invalid number of outputs!");
    exit(1);
  }

  logits = (float *)outputs[0].data;
  pred_boxes = (float *)outputs[1].data;

  for (int i = 0; i < 100; i++)
  {
    Mat raw_scores(1, class_name.size(), CV_32FC1, logits);
    Point class_id_point;
    double max_class_score;
    minMaxLoc(raw_scores, 0, &max_class_score, 0, &class_id_point);
    Mat scores = softmax(raw_scores, 0);
    minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

    if (max_class_score > score_threshold)
    {
      int cx     = pred_boxes[0] * x_factor;
      int cy     = pred_boxes[1] * y_factor;
      int width  = pred_boxes[2] * x_factor;
      int height = pred_boxes[3] * y_factor;

      int left = cx - width / 2;
      int top = cy - height / 2;

      class_ids.push_back(class_id_point.x);
      confidences.push_back(max_class_score);
      boxes.emplace_back(left, top, width, height);
    }

    logits += 92;
    pred_boxes += 4;
  }

  NMSBoxes(boxes, confidences, score_threshold, 0.1, indices);
  for (int i = 0; i < indices.size(); i++)
  {
    int idx = indices[i];
    detections.emplace_back(class_ids[idx], confidences[idx], boxes[idx]);
  }

  return detections;
}

int main(int argc, char **argv)
{

  // Initialize variables.
  Args args(argc, argv);

  Net net;

  FILE *pipein = NULL;
  char *frame_buffer = NULL;

  if ( class_list.empty() ) {
    if (args.yolov6 == 1)
      class_list = {
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
    else
      class_list = {
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", 
        "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", 
        "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", 
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "N/A",
        "backpack", "umbrella", "N/A", "N/A", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "N/A", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", 
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
        "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
        "dining table", "N/A", "N/A", "toilet", "N/A", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "N/A", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
      };
  }

  // Load model.
  net = readNet(args.model_path);
  if (net.empty())
  {
    printf("Error loading model!\n");
    return -1;
  }
  if (args.use_gpu)
  {
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);
  }

  // Open video stream.
  pipein = connect_to_video_host(args.host, args.port, args.use_gpu);
  if (pipein == NULL)
  {
    printf("Error connecting to video stream!\n");
    return -1;
  }

  // Allocate memory for frame buffer.
  FRAME_SIZE = args.original_image_width * args.original_image_height * 3;
  frame_buffer = (char *)malloc(FRAME_SIZE);
  if (frame_buffer == NULL)
  {
    printf("Error allocating memory for frame buffer!\n");
    return -1;
  }

  // Wait For the First Frame.
  size_t bytes = read_frame(pipein, frame_buffer);
  if (bytes < FRAME_SIZE)
  {
    printf("Error receiving first frame!\n");
    return -1;
  }
  printf("Connected! [%s:%s]\n", args.host.c_str(), args.port.c_str());
  thread t1(read_frames, pipein, frame_buffer);

  // Opencv Image around Frame Buffer.
  Mat frame(args.original_image_width, args.original_image_height, CV_8UC3, frame_buffer);
  printf("Frame size: %dx%d\n", frame.cols, frame.rows);

  // Initialize Gimbal.
  Gimbal gimbal(args.host);

  on_exit(stop, (void *) &gimbal);
  if (signal(SIGINT, cntl_c) == SIG_ERR)
    fprintf(stderr, "signal");

  // Main loop.
  double freq = getTickFrequency() / 1000;
  for (;;)
  {
    frame_mutex.lock();
    Mat image = frame.clone();
    frame_mutex.unlock();

    vector<Mat> outputs = pre_process(image, net, args.mean, args.standard_deviation, args.input_width, args.input_height);
    vector<Detection> detections;
    switch (args.yolov6)
    {
    case 0:
      detections = post_process_detr(outputs, class_list, net, args.score_threshold, 
        args.input_width, args.input_height, args.original_image_width, args.original_image_height);
      break;
    case 1:
      detections = post_process_yolo(outputs, class_list, net, args.score_threshold, 
        args.input_width, args.input_height, args.original_image_width, args.original_image_height);
      break;
    default:
      perror("Invalid model!");
      exit(1);
    }
    draw_detections(image, detections, class_list);

    vector<double> layersTimes;
    double t = net.getPerfProfile(layersTimes);
    cout << format("Frame [%d]:\t%.2f\tms", frame_no + 1, t / freq) << endl;

    cvtColor(image, image, COLOR_RGB2BGR);
    imshow("frame", image);
    if ((char)waitKey(1) == 'q')
      break;

    // Find Elephant in the room.
    int elephant_index = -1;
    for (int i = 0; i < class_list.size(); i++)
    {
      if (class_list[i] == "elephant")
      {
        elephant_index = i;
        break;
      }
    }

    // Find Elephant Detection in Detections.
  vector<Detection> elephant_detections;
  for (int i = 0; i < detections.size(); i++)
  {
    if (detections[i].class_id == elephant_index)
    {
      elephant_detections.push_back(detections[i]);
    }
  }

  // Find Elephant with highest confidence.
  int max_confidence_index = -1;
  float max_confidence = 0;
  for (int i = 0; i < elephant_detections.size(); i++)
  {
    if (elephant_detections[i].score > max_confidence)
    {
      max_confidence = elephant_detections[i].score;
      max_confidence_index = i;
    }
  }

  // If Elephant is detected, track it.
  if (max_confidence_index != -1)
  {
    Detection elephant_detection = elephant_detections[max_confidence_index];
    Rect elephant_box = elephant_detection.box;
    int elephant_x = elephant_box.x + elephant_box.width / 2;
    int elephant_y = elephant_box.y + elephant_box.height / 2;
    int x_error = elephant_x - image.cols / 2;
    int y_error = elephant_y - image.rows / 2;
    pid_loop(gimbal, 1300, 1300);
  }

  // Release resources.
  destroyAllWindows();
  t1.detach();
  return 0;
}