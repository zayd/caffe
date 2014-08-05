// Copyright 2014 BVLC and contributors.
#include <stdint.h>
#include <leveldb/db.h>
#include <pthread.h>
#include <cfloat>

#include <string>
#include <vector>
#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)
#include <iomanip>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::pair;

namespace caffe {

template <typename Dtype>
void* ImagePairDataLayerPrefetch(void* layer_pointer) {
  CHECK(layer_pointer);
  ImagePairDataLayer<Dtype>* layer =
      reinterpret_cast<ImagePairDataLayer<Dtype>*>(layer_pointer);
  CHECK(layer);
  Datum datum;
  CHECK(layer->prefetch_data_a_);
  CHECK(layer->prefetch_data_b_);
  Dtype* top_data_a = layer->prefetch_data_a_->mutable_cpu_data();
  Dtype* top_data_b = layer->prefetch_data_b_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  ImagePairDataParameter image_pair_data_param = layer->layer_param_.image_pair_data_param();
  const Dtype scale = image_pair_data_param.scale();
  const int batch_size = image_pair_data_param.batch_size();
  const int crop_size = image_pair_data_param.crop_size();
  const bool mirror = image_pair_data_param.mirror();
  const int new_height = image_pair_data_param.new_height();
  const int new_width = image_pair_data_param.new_width();

  if (mirror && crop_size == 0) {
    LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
        << "set at the same time.";
  }
  // datum scales
  const int channels = layer->channels_;
  const int height = layer->height_;
  const int width = layer->width_;
  const int size = layer->size_;
  const int lines_size = layer->lines_.size();
  const Dtype* mean = layer->data_mean_.cpu_data();
  string data_a;
  string data_b;
  for (int item_id = 0; item_id < batch_size; ++item_id) {
//    LOG(INFO) << "line " << layer->lines_id_;

    // get a blob
//    LOG(INFO) << "loading " << layer->lines_[layer->lines_id_].first.first;
    CHECK_GT(lines_size, layer->lines_id_);
    if (!ReadImageToData(layer->lines_[layer->lines_id_].first.first, data_a)) {
      continue;
    }
    CHECK_GT(data_a.size(), 0);

//    LOG(INFO) << "loading " << layer->lines_[layer->lines_id_].first.second;
    if (!ReadImageToData(layer->lines_[layer->lines_id_].first.second, data_b)) {
      continue;
    }
    CHECK_GT(data_b.size(), 0);

    for (int j = 0; j < size; ++j) {
      top_data_a[item_id * size + j] =
          (static_cast<Dtype>((uint8_t)data_a[j]) - mean[j]) * scale;

      top_data_b[item_id * size + j] =
          (static_cast<Dtype>((uint8_t)data_b[j]) - mean[j]) * scale;
    }

    top_label[item_id] = layer->lines_[layer->lines_id_].second;

    //LOG(INFO) << "lines_id_ " << layer->lines_id_;
    //LOG(INFO) << "item_id " << item_id << " Label " << top_label[item_id];

    // go to the next iter
    layer->lines_id_++;
    if (layer->lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
//      LOG(INFO) << "Restarting data prefetching from start.";
      layer->lines_id_ = 0;
      if (layer->layer_param_.image_pair_data_param().shuffle()) {
        LOG(INFO) << "Reshuffling.";
        std::random_shuffle(layer->lines_.begin(), layer->lines_.end());
      }
    }
    data_a.clear();
    data_b.clear();
  }

  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
ImagePairDataLayer<Dtype>::~ImagePairDataLayer<Dtype>() {
  // Finally, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
void ImagePairDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  //std::cout << "doing what";
  CHECK_EQ(bottom.size(), 0) << "Input Layer takes no input blobs.";
  CHECK_EQ(top->size(), 3) << "Input Layer takes three blobs as output.";
  const int new_height  = this->layer_param_.image_pair_data_param().new_height();
  const int new_width  = this->layer_param_.image_pair_data_param().new_height();
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_pair_data_param().source();
  const string& relative_path = this->layer_param_.image_pair_data_param().relative_path();
  // Add the relative path
  const string& relative_path_source = relative_path + source; 
  
  LOG(INFO) << "Opening file " << relative_path_source;
  std::ifstream infile(relative_path_source.c_str());

  int num_pos_pairs = 0;

  infile >> num_pos_pairs;

  string name_a;
  string name_b;
  int num_a;
  int num_b;
  int label = 1;

  // read the positive pairs ("same")
  for (int i = 0 ; i < num_pos_pairs ; i++) {

    infile >> name_a >> num_a >> num_b;

    std::ostringstream buff_a;
    buff_a << relative_path << "lfw/" << name_a << "/" << name_a << "_";
    buff_a << std::setfill('0') << std::setw(4) << num_a;
    buff_a << ".jpg";
    string filename_a = buff_a.str();
    
    std::ostringstream buff_b;
    buff_b << relative_path << "lfw/" << name_a << "/" << name_a << "_";
    buff_b << std::setfill('0') << std::setw(4) << num_b;
    buff_b << ".jpg";
    string filename_b = buff_b.str();

    //LOG(INFO) << "filename_a " << filename_a << " filename_b " << filename_b;

    lines_.push_back(
      std::make_pair(std::make_pair(filename_a, filename_b), label)
    );
  }

  // the rest are negative pairs ("not same")
  label = -1;
  while (infile >> name_a >> num_a >> name_b >> num_b) {

    std::ostringstream buff_a;
    buff_a << relative_path << "lfw/" << name_a << "/" << name_a << "_";
    //buff_a << "lfw/" << name_a << "/" << name_a << "_";
    buff_a << std::setfill('0') << std::setw(4) << num_a;
    buff_a << ".jpg";
    string filename_a = buff_a.str();

    std::ostringstream buff_b;
    buff_b << relative_path << "lfw/" << name_b << "/" << name_b << "_";
    //buff_b << "lfw/" << name_b << "/" << name_b << "_";
    buff_b << std::setfill('0') << std::setw(4) << num_b;
    buff_b << ".jpg";
    string filename_b = buff_b.str();

    //LOG(INFO) << "filename_a " << filename_a << " filename_b " << filename_b;

    lines_.push_back(
      std::make_pair(std::make_pair(filename_a, filename_b), label)
    );
  }

  if (this->layer_param_.image_pair_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    std::random_shuffle(lines_.begin(), lines_.end());
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_pair_data_param().rand_skip()) {
    // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
    unsigned int skip = rand() %
        this->layer_param_.image_pair_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enought points to skip";
    lines_id_ = skip;
  }
  // Read a data point, and use it to initialize the top blob.
  Datum datum;
  CHECK(ReadImageToDatum(lines_[lines_id_].first.first, lines_[lines_id_].second,
                         new_height, new_width, &datum));

  // image
  const int crop_size = this->layer_param_.image_pair_data_param().crop_size();
  const int batch_size = this->layer_param_.image_pair_data_param().batch_size();
  const string& mean_file = this->layer_param_.image_pair_data_param().mean_file();
  const string& relative_path_mean_file = relative_path + mean_file;
  if (crop_size > 0) {
    (*top)[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    prefetch_data_a_.reset(new Blob<Dtype>(batch_size, datum.channels(),
                                         crop_size, crop_size));

    (*top)[1]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
    prefetch_data_b_.reset(new Blob<Dtype>(batch_size, datum.channels(),
                                             crop_size, crop_size));
  } else {
    (*top)[0]->Reshape(batch_size, datum.channels(), datum.height(),
                       datum.width());
    prefetch_data_a_.reset(new Blob<Dtype>(batch_size, datum.channels(),
                                            datum.height(), datum.width()));
    (*top)[1]->Reshape(batch_size, datum.channels(), datum.height(),
                           datum.width());
    prefetch_data_b_.reset(new Blob<Dtype>(batch_size, datum.channels(),
                                         datum.height(), datum.width()));
  }
  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();


  // label
  (*top)[2]->Reshape(batch_size, 1, 1, 1);

  prefetch_label_.reset(new Blob<Dtype>(batch_size, 1, 1, 1));
  // datum size
  channels_ = datum.channels();
  height_ = datum.height();
  width_ = datum.width();
  size_ = datum.channels() * datum.height() * datum.width();
  CHECK_GT(height_, crop_size);
  CHECK_GT(width_, crop_size);

  // check if we want to have mean
  if (this->layer_param_.image_pair_data_param().has_mean_file()) {
    BlobProto blob_proto; 
    LOG(INFO) << "Loading mean file from" << relative_path_mean_file;
    ReadProtoFromBinaryFile(relative_path_mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.channels(), channels_);
    CHECK_EQ(data_mean_.height(), height_);
    CHECK_EQ(data_mean_.width(), width_);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, channels_, height_, width_);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_a_->mutable_cpu_data();
  prefetch_data_b_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CHECK(!pthread_create(&thread_, NULL, ImagePairDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
Dtype ImagePairDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_a_->cpu_data(),
      sizeof(Dtype) * prefetch_data_a_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_data_b_->cpu_data(),
        sizeof(Dtype) * prefetch_data_b_->count());
  memcpy((*top)[2]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());

  Dtype val, sum, min, max;
  int count = prefetch_data_a_->count();
  for (int j = 0 ; j < 2 ; j++) {
    sum = 0.0;
    min = FLT_MAX;
    max = -FLT_MAX;
    for (int i = 0 ; i < count ; i++) {
      val = (*top)[j]->mutable_cpu_data()[i];
      sum += val;
      if (val < min) min = val;
      if (val > max) max = val;
    }
//    printf("data %d min/max/avg: %.4f/%.4f/%.4f\n", j, min, max, sum / count);
  }

  sum = 0.0;
  min = FLT_MAX;
  max = -FLT_MAX;

  for (int i = 0 ; i < prefetch_label_->count() ; i++) {
    val = (*top)[2]->mutable_cpu_data()[i];
    sum += val;
    if (val < min) min = val;
    if (val > max) max = val;
  }
//  printf("label min/max/avg: %.4f/%.4f/%.4f\n", min, max, sum / count);

  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, ImagePairDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImagePairDataLayer);

}  // namespace caffe
