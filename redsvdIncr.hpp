/* 
 *  Copyright (c) 2011 Daisuke Okanohara
 * 
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 * 
 *   1. Redistributions of source code must retain the above Copyright
 *      notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above Copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *   3. Neither the name of the authors nor the names of its contributors
 *      may be used to endorse or promote products derived from this
 *      software without specific prior written permission.
 */

#ifndef REDSVD_INCR_HPP__
#define REDSVD_INCR_HPP__

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET

#include <vector>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include "util.hpp"

namespace REDSVD {

class RedSVDIncr {
public:
  RedSVDIncr(){}

  template <class Reader>
  RedSVDIncr(Reader& reader, const int rank){
    run(reader, rank);
  }

  template <class Reader>
  void run(Reader& reader, const int rank){
    int r = (rank < reader.cols()) ? rank : reader.cols();
    Eigen::MatrixXd O(reader.rows(), r);
    Util::sampleGaussianMat(O);

    Eigen::MatrixXd Y = Eigen::MatrixXd::Zero(reader.cols(), r);
    for (int row = 0; row < reader.rows(); ++row){
      fv_t fv;
      reader.ReadRow(fv);
      for (size_t i = 0; i < fv.size(); ++i){
        int column = fv[i].first;
        double val = fv[i].second;
        for (int j = 0; j < r; ++j){
          Y(column, j) += O(row, j) * val;
        }
      }
    }
    Util::processGramSchmidt(Y);

    reader.Rewind();
    
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(reader.rows(), r);
    for (int row = 0; row < reader.rows(); ++row){
      fv_t fv;
      reader.ReadRow(fv);
      for (size_t i = 0; i < fv.size(); ++i){
        int column = fv[i].first;
        double val = fv[i].second;
        for (int j = 0; j < r; ++j){
          B(row, j) += val * Y(column, j);
        }
      }
    }

    // Gaussian Random Matrix
    Eigen::MatrixXd P(B.cols(), r);
    Util::sampleGaussianMat(P);
    
    // Compute Sample Matrix of B
    Eigen::MatrixXd Z = B * P;
    
    // Orthonormalize Z
    Util::processGramSchmidt(Z);
    
    // Range(C) = Range(B)
    Eigen::MatrixXd C = Z.transpose() * B; 
    
    Eigen::JacobiSVD<Eigen::MatrixXd> svdOfC(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    // C = USV^T
    // A = Z * U * S * V^T * Y^T()
    matU_ = Z * svdOfC.matrixU();
    matS_ = svdOfC.singularValues();
    matV_ = Y * svdOfC.matrixV();
  }
  
  const Eigen::MatrixXd& matrixU() const {
    return matU_;
  }

  const Eigen::VectorXd& singularValues() const {
    return matS_;
  }

  const Eigen::MatrixXd& matrixV() const {
    return matV_;
  }

private:
  Eigen::MatrixXd matU_;
  Eigen::VectorXd matS_;
  Eigen::MatrixXd matV_;
};

}

#endif // REDSVD_INCR_HPP__
