#include <algorithm>
#include <vector>

#include "caffe/ex_layers/xnor_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define sign(x) ((x)>=0?1:-1)

template <typename Dtype>
void XnorNetLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(top.size(), 1);
  CHECK_EQ(bottom.size(), 1);

  //BinActiv SetUp
  LayerParameter binact_param(this->layer_param_);
  binact_param.set_type("BinActiv");
  binact_param.mutable_convolution_param()->CopyFrom(
    this->layer_param_.convolution_param() );
  binactiv_layer_ = LayerRegistry<Dtype>::CreateLayer(binact_param);
  binactiv_top_vec_.clear();
  binactiv_top_B.reset(new Blob<Dtype>());
  binactiv_top_K.reset(new Blob<Dtype>());
  binactiv_top_vec_.push_back(binactiv_top_B.get());
  binactiv_top_vec_.push_back(binactiv_top_K.get());
  binactiv_layer_->SetUp(bottom, binactiv_top_vec_);
  DLOG(INFO) << "XnorNetLayer[ " << this->layer_param_.name() << " ] BinActiv SetUp";

  //SplitConcat SetUp
  LayerParameter splitconcat_param(this->layer_param_);
  splitconcat_param.set_type("SplitConcat");
  splitconcat_param.mutable_convolution_param()->CopyFrom(
    this->layer_param_.convolution_param() );
  splitconcat_layer_ = LayerRegistry<Dtype>::CreateLayer(splitconcat_param);
  splitconcat_bottom_vec_.clear();
  splitconcat_bottom_vec_.push_back(binactiv_top_vec_[1]);
  splitconcat_top_shared_.reset(new Blob<Dtype>());
  splitconcat_top_vec_.clear();
  splitconcat_top_vec_.push_back(splitconcat_top_shared_.get());
  splitconcat_layer_->SetUp(splitconcat_bottom_vec_, splitconcat_top_vec_);
  DLOG(INFO) << "XnorNetLayer[ " << this->layer_param_.name() << " ] SplitConcat SetUp";

  //BinaryConvolution
  LayerParameter binconv_param(this->layer_param_);
  binconv_param.set_type("BinaryConvolution");
  binconv_param.mutable_convolution_param()->CopyFrom(
    this->layer_param_.convolution_param() );
  binaryconvolution_layer_ = LayerRegistry<Dtype>::CreateLayer(binconv_param);
  binaryconvolution_bottom_vec_.clear();
  binaryconvolution_bottom_vec_.push_back(binactiv_top_B.get());
  binaryconvolution_top_vec_shared_.reset(new Blob<Dtype>());
  binaryconvolution_top_vec_.clear();
  binaryconvolution_top_vec_.push_back(binaryconvolution_top_vec_shared_.get());
  binaryconvolution_layer_->SetUp(binaryconvolution_bottom_vec_, binaryconvolution_top_vec_);
  DLOG(INFO) << "XnorNetLayer[ " << this->layer_param_.name() << " ] BinaryConvolution SetUp";
  //For Share Convolution Parameters with BinaryConvolution Layer
  const vector<shared_ptr<Blob<Dtype> > >& binconv_params = binaryconvolution_layer_->blobs();
  this->blobs_.resize( binconv_params.size() );
  
  for (size_t index = 0; index < this->blobs_.size(); index++ ) {
    this->blobs_[index].reset(new Blob<Dtype>());
    //this->blobs_[index]->CopyFrom(*(binconv_params[index].get()), false, true);
    //DLOG(INFO) << "XnorNetLayer BinaryConvolution : param_size : " << binconv_params.size() << " [0] : " << binconv_params[0]->num() << ", " << binconv_params[0]->channels() << ", " << binconv_params[0]->height() << ", " << binconv_params[0]->width();
    this->blobs_[index]->Reshape( binconv_params[index]->shape() );
    this->blobs_[index]->ShareData( *(binconv_params[index].get()) );
    this->blobs_[index]->ShareDiff( *(binconv_params[index].get()) );
    //binconv_params[index]->ShareData( *(this->blobs_[index].get()) );
    //binconv_params[index]->ShareDiff( *(this->blobs_[index].get()) );
  }
  DLOG(INFO) << "XnorNetLayer[ " << this->layer_param_.name() << " ] Share Params SetUp";

  //Eltwise
  LayerParameter eltwise_param(this->layer_param_);
  eltwise_param.set_type("Eltwise");
  eltwise_param.mutable_convolution_param()->CopyFrom(
    this->layer_param_.convolution_param() );
  ::caffe::EltwiseParameter* eltwise_cur = eltwise_param.mutable_eltwise_param();
  eltwise_cur->set_operation( EltwiseParameter_EltwiseOp_PROD );
  eltwise_layer_ = LayerRegistry<Dtype>::CreateLayer(eltwise_param); 
  eltwise_bottom_vec_.clear();
  eltwise_bottom_vec_.push_back(binaryconvolution_top_vec_shared_.get());
  eltwise_bottom_vec_.push_back(splitconcat_top_shared_.get());
  eltwise_layer_->SetUp(eltwise_bottom_vec_, top);

  DLOG(INFO) << "XnorNetLayer[ " << this->layer_param_.name() << " ] Eltwise SetUp";
}

template <typename Dtype>
void XnorNetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  binactiv_layer_->Reshape(bottom, binactiv_top_vec_);
  splitconcat_layer_->Reshape(splitconcat_bottom_vec_, splitconcat_top_vec_);
  binaryconvolution_layer_->Reshape(binaryconvolution_bottom_vec_, binaryconvolution_top_vec_);
  eltwise_layer_->Reshape(eltwise_bottom_vec_, top);
}

template <typename Dtype>
void XnorNetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  binactiv_layer_->Forward(bottom, binactiv_top_vec_);
  splitconcat_layer_->Forward(splitconcat_bottom_vec_, splitconcat_top_vec_);
  
  //In Case In Train Solver , The weights shared by Test Net and Train Net
  if (this->phase_ == TEST) {
    const vector<shared_ptr<Blob<Dtype> > >& binconv_params = binaryconvolution_layer_->blobs();
    for (size_t index = 0; index < this->blobs_.size(); index++ ) {
      binconv_params[index]->CopyFrom(*(this->blobs_[index].get()), false, false);
    }
  }
  binaryconvolution_layer_->Forward(binaryconvolution_bottom_vec_, binaryconvolution_top_vec_);
  eltwise_layer_->Forward(eltwise_bottom_vec_, top);
}

template <typename Dtype>
void XnorNetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if ( propagate_down[0] == false ) return;
  vector<bool> eltwise_propagate_down(2,false);
  eltwise_propagate_down[0] = true;
  eltwise_layer_->Backward(top, eltwise_propagate_down, eltwise_bottom_vec_);
  binaryconvolution_layer_->Backward(binaryconvolution_top_vec_, propagate_down, binaryconvolution_bottom_vec_);
  binactiv_layer_->Backward(binactiv_top_vec_, eltwise_propagate_down, bottom);
}

INSTANTIATE_CLASS(XnorNetLayer);
REGISTER_LAYER_CLASS(XnorNet);

}  // namespace caffe
