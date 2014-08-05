// Copyright 2014 BVLC and contributors.

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void BinaryAccuracyLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Accuracy Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Accuracy Layer takes 1 output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
  (*top)[0]->Reshape(1, 1, 1, 1); 
}

template <typename Dtype>
Dtype BinaryAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();  
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    // Accuracy
    if ( bottom_data[i] * bottom_label[i] > Dtype(0.0) ) {
      accuracy++;
    }
  }
  // LOG(INFO) << "Accuracy: " << accuracy;
  (*top)[0]->mutable_cpu_data()[0] = accuracy / num;
  return Dtype(0);
}

INSTANTIATE_CLASS(BinaryAccuracyLayer);

}  // namespace caffe
