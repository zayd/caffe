// Copyright 2014 BVLC and contributors.

#include <cuda_runtime.h>

#include <iostream>  // NOLINT(readability/streams)
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <unistd.h>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/test/test_caffe_main.hpp"

using std::string;

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class ImagePairDataLayerTest : public ::testing::Test {
 protected:
	ImagePairDataLayerTest()
      : blob_top_data_a_(new Blob<Dtype>()),
        blob_top_data_b_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_a_);
    blob_top_vec_.push_back(blob_top_data_b_);
    blob_top_vec_.push_back(blob_top_label_);
  }

  virtual ~ImagePairDataLayerTest() {
    delete blob_top_data_a_;
    delete blob_top_data_b_;
    delete blob_top_label_;
  }

  Blob<Dtype>* const blob_top_data_a_;
  Blob<Dtype>* const blob_top_data_b_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ImagePairDataLayerTest, Dtypes);

TYPED_TEST(ImagePairDataLayerTest, TestRead) {
  LayerParameter param;
  ImagePairDataParameter* image_pair_data_param = param.mutable_image_pair_data_param();
  image_pair_data_param->set_batch_size(1);
  image_pair_data_param->set_relative_path("../../examples/lfw/");
  image_pair_data_param->set_source("pairsDevTrain.txt");
  image_pair_data_param->set_shuffle(false);
  ImagePairDataLayer<TypeParam> layer(param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_a_->num(), 1);
  EXPECT_EQ(this->blob_top_data_a_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_a_->height(), 50);
  EXPECT_EQ(this->blob_top_data_a_->width(), 50);
  EXPECT_EQ(this->blob_top_data_b_->num(), 1);
  EXPECT_EQ(this->blob_top_data_b_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_b_->height(), 50);
  EXPECT_EQ(this->blob_top_data_b_->width(), 50);
  EXPECT_EQ(this->blob_top_label_->num(), 1);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 1);
  EXPECT_EQ(this->blob_top_label_->width(), 1);

  for (int iter = 0; iter < 2200; ++iter) {
    layer.Forward(this->blob_bottom_vec_, &this->blob_top_vec_);
    if (iter < 1100)
    {
      EXPECT_EQ((int)(*this->blob_top_label_->cpu_data()), 1);
    }
    else
    {
      EXPECT_EQ((int)(*this->blob_top_label_->cpu_data()), -1);
    }
  }
}

}  // namespace caffe
