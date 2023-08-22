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

int YOLOV6 = 1;
float INPUT_WIDTH = 640.0;
float INPUT_HEIGHT = 640.0;
float IMAGE_WIDTH = 1280.0;
float IMAGE_HEIGHT = 1280.0;
double SCALE_FACTOR = 1.0;
Scalar MEAN = Scalar(0.0, 0.0, 0.0);
float standard_deviation[3] = {1., 1., 1.};
Mat stdv = Mat(1, 3, CV_32FC1, standard_deviation);

float SCORE_THRESHOLD = 0.5;
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
int FRAME_SIZE = (int)IMAGE_WIDTH * (int)IMAGE_HEIGHT * 3;
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

char *parse_args(int argc, char **argv, Net &net, string &host, string &port, FILE *&pipein, vector<string> &class_list)
{
  const char *usage = "Run Yolov6 3.0 or Facebook's DETR.\nConnect to [host]:5001 and read a H264 encoded video stream.\n\n"
                      "Usage: %s [options]\n\n"
                      "Options:\n"
                      "    -m, --model <path>                             Path to ONNX model.\n"
                      "    -c, --classname-file <path>                    Path to class name file (e.g coco.names).\n"
                      "    -g, --use-gpu <bool>                           Use GPU or not. (default: true  [1])\n"
                      "    -a, --host|address <address>                   Host address. (default: gabriel.local)\n"
                      "    -s, --original-image-size <width>x<height>     Original image size. (default: 1280x1280)\n"
                      "    -S, --input-image-size <width>x<height>        ONNX model's input image size. (default: 640x640)\n"
                      "    -t, --score-threshold <float>                  Score threshold. (default: 0.5)\n"
                      "    -m, --mean <float>,<float>,<float>             Mean values. Comma seperated, no whitespace. (default: 0,0,0)\n"
                      "    -d, --stdv <float>,<float>,<float>              Standard deviation. (default: 1,1,1)\n"
                      "    -y  --yolov6 <int>                             [Yolov6] detection output format. (default: true [1])\n"
                      "    -h, --help                                     Print this message.\n";

  ifstream classfile_handle;
  string classfile_line;
  string model_path;
  bool use_gpu = true;

  int opt;

  while (1)
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
            {"stdv", required_argument, 0, 'd'},
            {"standard-deviation", required_argument, 0, 'd'},
            {"yolov6", required_argument, 0, 'y'},
            {"help", no_argument, 0, 'h'},
            {0, 0, 0, 0}};
    opt = getopt_long(argc, argv, "m:c:g:a:p:s:S:t:k:d:h",
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

    case 'c':
    case 'n':
      class_list.clear();
      classfile_handle.open(optarg);
      while (getline(classfile_handle, classfile_line))
      {
        class_list.push_back(classfile_line);
      }
      break;

    case 'g':
      use_gpu = atoi(optarg);
      break;

    case 's':
      if (sscanf(optarg, "%fx%f", &IMAGE_WIDTH, &IMAGE_HEIGHT) != 2)
      {
        fprintf(stderr, "Invalid image size!");
        fprintf(stderr, usage, argv[0]);
        exit(1);
      }
      FRAME_SIZE = (int)IMAGE_WIDTH * (int)IMAGE_HEIGHT * 3;
      break;

    case 'S':
      if (sscanf(optarg, "%fx%f", &INPUT_WIDTH, &INPUT_HEIGHT) != 2)
      {
        fprintf(stderr, "Invalid model input size!");
        fprintf(stderr, usage, argv[0]);
        exit(1);
      }
      break;

    case 't':
      if (sscanf(optarg, "%f", &SCORE_THRESHOLD) != 1)
      {
        fprintf(stderr, "Invalid score threshold!");
        fprintf(stderr, usage, argv[0]);
        exit(1);
      }
      break;

    case 'k':
      if (sscanf(optarg, "%lf,%lf,%lf", &MEAN[0], &MEAN[1], &MEAN[2]) != 3)
      {
        fprintf(stderr, "Invalid mean value!");
        fprintf(stderr, usage, argv[0]);
        exit(1);
      }
      break;

    case 'd':
      if (sscanf(optarg, "%f,%f,%f", &standard_deviation[0], &standard_deviation[1], &standard_deviation[2]) != 3)
      {
        fprintf(stderr, "Invalid standard deviation!\n");
        fprintf(stderr, usage, argv[0]);
        exit(1);
      }
      break;

    case 'y':
      YOLOV6 = atoi(optarg);
      break;

    case 'h':
      printf(usage, argv[0]);
      exit(0);

    case '?':
      fprintf(stderr, usage, argv[0]);
      exit(1);

    case ':':
      fprintf(stderr, usage, argv[0]);
      exit(1);

    default:
      abort();
    }
  }

  // Frame Buffer.
  // I could probably use RAII here, but this is faster.
  // We need the buffer for the entire duration of the program anyway...
  // Just need to make sure we call this function only once.
  char *frame_buffer = (char *)malloc(FRAME_SIZE);
  if (frame_buffer == NULL)
  {
    perror("Error allocating frame buffer!\n");
    exit(1);
  }

  const char *ffmpeg_cmd_format =
      "ffmpeg -nostdin -probesize 32 -flags low_delay -fflags nobuffer "
      "-codec:v %s -r 25 -i tcp://%s:%s "
      "-pix_fmt rgb24 -an -vcodec rawvideo -f rawvideo pipe: 2>/dev/null";

  const char *codec = use_gpu ? "h264_cuvid" : "h264";

  int size_s = snprintf(nullptr, 0, ffmpeg_cmd_format, codec, host.c_str(), port.c_str()) + 1;
  if (size_s <= 0)
  {
    perror("Error during formatting.");
    exit(1);
  }
  auto size = static_cast<size_t>(size_s);
  char ffmpeg_cmd[size];
  snprintf(ffmpeg_cmd, size, ffmpeg_cmd_format, codec, host.c_str(), port.c_str());

  pipein = popen(ffmpeg_cmd, "r");
  if (pipein == NULL)
  {
    perror("Error opening video stream!\n");
    exit(1);
  }

  net = readNetFromONNX(model_path.empty() ? "../yolov6.onnx" : model_path);
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

  vector<String> outNames = net.getUnconnectedOutLayersNames();
  vector<int> outLayers = net.getUnconnectedOutLayers();
  for (int i = 0; i < outNames.size(); i++)
  {
    printf("Output Layer [%d]: %s\n", outLayers[i], outNames[i].c_str());

    MatShape netInputShape{1, 3, (int)INPUT_WIDTH, (int)INPUT_HEIGHT};
    vector<MatShape> inLayerShape, outLayerShape;
    net.getLayerShapes(netInputShape, outLayers[i], inLayerShape, outLayerShape);

    printf("Output Layer Shape: ");
    for (int j = 0; j < outLayerShape.size(); j++)
    {
      for (int k = 0; k < outLayerShape[j].size(); k++)
        printf("%d (%d, %d), ", outLayerShape[j][k], j, k);
    }
    printf("\n");
  }

  printf("Loaded model! [%s]\n", model_path.empty() ? "../yolov6.onnx" : model_path.c_str());

  return frame_buffer;
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

vector<Mat> pre_process(InputArray image, Net &net)
{
  Mat blob;

  float first_pixel = image.getMat().ptr<float>(0, 0)[0];

  blobFromImage(image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), MEAN * 255.0, true, false);

  float first_pixel_before = blob.ptr<float>(0, 0)[0];

  for (int i = 0; i < blob.size[0]; ++i){
    float *data = blob.ptr<float>(0, i);
    for (int j = 0; j < blob.size[2] * blob.size[3]; ++j){
      data[j] = data[j] / standard_deviation[j % 3];
    }
  }

  float first_pixel_after = blob.ptr<float>(0, 0)[0];

  printf("Mean: %f, %f, %f\n", MEAN[0], MEAN[1], MEAN[2]);
  printf("Standard Deviation: %f, %f, %f\n", standard_deviation[0], standard_deviation[1], standard_deviation[2]);
  printf("First pixel: %f\n", first_pixel);
  printf("First pixel before: %f\n", first_pixel_before);
  printf("First pixel after: %f\n", first_pixel_after);

  net.setInput(blob);

  vector<Mat> outputs;
  net.forward(outputs, net.getUnconnectedOutLayersNames());

  return outputs;
}

vector<Detection> post_process_yolo(vector<Mat> &outputs, const vector<string> &class_name, Net &net)
{
  // Initialize vectors to hold respective outputs while unwrapping detections.
  vector<int> class_ids;
  vector<float> confidences;
  vector<float> cls_scores;
  vector<Rect> boxes;

  // Resizing factor.
  float x_factor = IMAGE_WIDTH / INPUT_WIDTH;
  float y_factor = IMAGE_HEIGHT / INPUT_HEIGHT;

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
      if (max_class_score > SCORE_THRESHOLD)
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

  printf("Number of detections: %ld\n", boxes.size());

  // Perform Non Maximum Suppression and draw predictions.
  vector<int> indices;
  NMSBoxes(boxes, cls_scores, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

  vector<Detection> Detections;
  for (int i = 0; i < indices.size(); i++)
  {
    int idx = indices[i];
    Detections.emplace_back(class_ids[idx], cls_scores[idx], boxes[idx]);
  }

  printf("Number of detections after NMS: %ld\n", Detections.size());
  return Detections;
}

vector<Detection> post_process_detr(vector<Mat> &outputs, const vector<string> &class_name, Net &net)
{
  vector<Rect> boxes;
  vector<float> confidences;
  vector<int> indices;
  vector<int> class_ids;

  vector<Detection> detections;

  float x_factor = IMAGE_WIDTH;
  float y_factor = IMAGE_HEIGHT;

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

    if (max_class_score > SCORE_THRESHOLD)
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

  printf("Number of detections: %ld\n", boxes.size());

  NMSBoxes(boxes, confidences, SCORE_THRESHOLD, 0.1, indices);
  for (int i = 0; i < indices.size(); i++)
  {
    int idx = indices[i];
    detections.emplace_back(class_ids[idx], confidences[idx], boxes[idx]);
  }

  printf("Number of detections after NMS: %ld\n", detections.size());
  return detections;
}

int main(int argc, char **argv)
{

  // Initialize variables.
  Net net;
  string host = "gabriel.local";
  string port = "5001";
  FILE *pipein = NULL;
  char *frame_buffer = NULL;

  frame_buffer = parse_args(argc, argv, net, host, port, pipein, class_list);
  if ( class_list.empty() ) {
    if (YOLOV6 == 1)
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

  // Wait for the first frame.
  size_t bytes = read_frame(pipein, frame_buffer);
  if (bytes < FRAME_SIZE)
  {
    printf("Error receiving first frame!\n");
    return -1;
  }
  printf("Connected! [%s:%s]\n", host.c_str(), port.c_str());
  thread t1(read_frames, pipein, frame_buffer);

  // Opencv Image Buffer.
  Mat frame(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC3, frame_buffer);

  // Main loop.
  double freq = getTickFrequency() / 1000;
  for (;;)
  {
    frame_mutex.lock();
    Mat image = frame.clone();
    frame_mutex.unlock();

    vector<Mat> outputs = pre_process(image, net);
    vector<Detection> detections;
    switch (YOLOV6)
    {
    case 0:
      detections = post_process_detr(outputs, class_list, net);
      break;
    case 1:
      detections = post_process_yolo(outputs, class_list, net);
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
  }

  // Release resources.
  destroyAllWindows();
  t1.detach();
  return 0;
}