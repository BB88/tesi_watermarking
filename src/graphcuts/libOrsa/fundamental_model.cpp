/**
 * @file fundamental_model.cpp
 * @brief Fundamental matrix model
 * @author Pascal Monasse, Pierre Moulon
 * 
 * Copyright (c) 2011 Pascal Monasse, Pierre Moulon
 * All rights reserved.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define _USE_MATH_DEFINES // For Windows
#include "../libOrsa/fundamental_model.hpp"
#include "../libOrsa/conditioning.hpp"
#include "../libNumerics/numerics.h"

namespace orsa {

/// Constructor, computing logalpha0_
FundamentalModel::FundamentalModel(const Mat &x1, int w1, int h1,
                                   const Mat &x2, int w2, int h2,
                                   bool symError)
: OrsaModel(x1, w1, h1, x2, w2, h2), symError_(symError) {
  double D, A; // Diameter and area of image
  D = sqrt(w1*(double)w1 + h1*(double)h1);
  A = w1*(double)h1;
  logalpha0_[0] = log10(2.0*D/A /N1_(0,0));
  D = sqrt(w2*(double)w2 + h2*(double)h2);
  A = w2*(double)h2;
  logalpha0_[1] = log10(2.0*D/A /N2_(0,0));
}

/// Unnormalize a given model (from normalized to image space).
void FundamentalModel::Unnormalize(Model * model) const  {
  UnnormalizerT::Unnormalize(N1_, N2_, model);
}

/**
 * Build a 9 x n matrix from point matches, where each row is equivalent to the
 * equation x'T*F*x = 0 for a single correspondence pair (x', x). The domain of
 * the matrix is a 9 element vector corresponding to F. In other words, set up
 * the linear system
 *
 *   Af = 0,
 *
 * where f is the F matrix as a 9-vector rather than a 3x3 matrix (row
 * major). If the points are well conditioned and there are 8 or more, then
 * the nullspace should be rank one. If the nullspace is two dimensional,
 * then the rank 2 constraint must be enforced to identify the appropriate F
 * matrix.
 *
 * Note that this does not resize the matrix A; it is expected to have the
 * appropriate size already.
 */
static void EncodeEpipolarEquation(const OrsaModel::Mat &x1,
                                   const OrsaModel::Mat &x2,
                                   const std::vector<int> &indices,
                                   OrsaModel::Mat *A) {
  for (size_t i = 0; i < indices.size(); ++i) {
    int j = indices[i];
    (*A)(i, 0) = x2(0,j) * x1(0,j);  // 0 represents x coords,
    (*A)(i, 1) = x2(0,j) * x1(1,j);  // 1 represents y coords.
    (*A)(i, 2) = x2(0,j);
    (*A)(i, 3) = x2(1,j) * x1(0,j);
    (*A)(i, 4) = x2(1,j) * x1(1,j);
    (*A)(i, 5) = x2(1,j);
    (*A)(i, 6) = x1(0,j);
    (*A)(i, 7) = x1(1,j);
    (*A)(i, 8) = 1.0;
  }
}

/// Find coefficients of polynomial det(F1+xF2), F1 and F2 being 3x3.
/// The polynomial is written a[0]+a[1]X+a[2]X^2+a[3]X^3.
static void det(const OrsaModel::Mat& F1,const OrsaModel::Mat& F2,double a[4]) {
    assert(F1.nrow()==3 && F1.ncol()==3 && F2.nrow()==3 && F2.ncol()==3);
    a[0] = a[1] = a[2] = a[3] = 0.0f;
    for(int i0=0; i0 < 3; i0++) { // Even permutations
        int i1 = (i0+1)%3;
        int i2 = (i1+1)%3;
        a[0] += F1(i0,0)*F1(i1,1)*F1(i2,2);
        a[1] += F2(i0,0)*F1(i1,1)*F1(i2,2)+
                F1(i0,0)*F2(i1,1)*F1(i2,2)+
                F1(i0,0)*F1(i1,1)*F2(i2,2);
        a[2] += F1(i0,0)*F2(i1,1)*F2(i2,2)+
                F2(i0,0)*F1(i1,1)*F2(i2,2)+
                F2(i0,0)*F2(i1,1)*F1(i2,2);
        a[3] += F2(i0,0)*F2(i1,1)*F2(i2,2);
    }
    for(int i0=0; i0 < 3; i0++) { // Odd permutations
        int i1 = (i0+2)%3;
        int i2 = (i1+2)%3;
        a[0] -= F1(i0,0)*F1(i1,1)*F1(i2,2);
        a[1] -= F2(i0,0)*F1(i1,1)*F1(i2,2)+
                F1(i0,0)*F2(i1,1)*F1(i2,2)+
                F1(i0,0)*F1(i1,1)*F2(i2,2);
        a[2] -= F1(i0,0)*F2(i1,1)*F2(i2,2)+
                F2(i0,0)*F1(i1,1)*F2(i2,2)+
                F2(i0,0)*F2(i1,1)*F1(i2,2);
        a[3] -= F2(i0,0)*F2(i1,1)*F2(i2,2);
    }
}
// Compute the real roots of a third order polynomial.
// Return the number of roots found (1 or 3).
static int cubicRoots(double coeff[4], double x[3]) {
    double a1 = coeff[2] / coeff[3];
    double a2 = coeff[1] / coeff[3];
    double a3 = coeff[0] / coeff[3];

    double Q = (a1 * a1 - 3 * a2) / 9;
    double R = (2 * a1 * a1 * a1 - 9 * a1 * a2 + 27 * a3) / 54;
    double Q3 = Q * Q * Q;
    double d = Q3 - R * R;

    if(d >= 0) { // 3 real roots
        double theta = acos(R / sqrt(Q3));
        double sqrtQ = sqrt(Q);
        x[0] = -2*sqrtQ*cos( theta          /3) - a1/3;
        x[1] = -2*sqrtQ*cos((theta + 2*M_PI)/3) - a1/3;
        x[2] = -2*sqrtQ*cos((theta + 4*M_PI)/3) - a1/3;
        return 3;
    } else { // 1 real root
        double e = pow(sqrt(-d) + fabs(R), 1.0/3.0);
        if(R > 0)
            e = -e;
        x[0] = static_cast<float>((e + Q / e) - a1 / 3.);
        return 1;
    }
}

void FundamentalModel::Fit(const std::vector<int> &indices,
                           std::vector<Mat> *Fs) const {
  assert(2 == x1_.nrow());
  assert(7 <= x1_.ncol());
  assert(x1_.nrow() == x2_.nrow());
  assert(x1_.ncol() == x2_.ncol());

  // Set up the homogeneous system Af = 0 from the equations x'T*F*x = 0.
  Mat A(indices.size(), 9);
  EncodeEpipolarEquation(x1_, x2_, indices, &A);

  if(indices.size() >= 8) { // 8-point algorithm
    // Without constraint
    libNumerics::vector<double> vecNullspace(9);
    libNumerics::SVD::Nullspace(A, &vecNullspace, 1, 1);
    libNumerics::matrix<double> F(3,3);
    F.read(vecNullspace);

    // Force the fundamental property if the A matrix has full rank.
    libNumerics::matrix<double> FRank2(3,3);
    libNumerics::SVD::EnforceRank2_3x3(F, &FRank2);
    Fs->push_back(FRank2);
  }
  else
  {
    // Find the two F matrices in the nullspace of A.
    Mat F1(3,3), F2(3,3);
    libNumerics::SVD::Nullspace2_Remap33(A,F1,F2);

    // Then, use the condition det(F) = 0 to determine F. In other words, solve
    // det(F1 + x*F2) = 0 for x.
    double a[] = {0,0,0,0};
    det(F1,F2,a);

    // Find the roots of the polynomial
    double roots[3];
    int num_roots = cubicRoots(a, roots);

    // Build the fundamental matrix for each solution.
    for (int s = 0; s < num_roots; ++s)
      Fs->push_back(F1 + roots[s] * F2);
  }
}

/// \param F The fundamental matrix.
/// \param index The point correspondence.
/// \param side In which image is the error measured?
/// \return The square reprojection error.
double FundamentalModel::Error(const Mat &F, int index, int* side) const {
  double xa = x1_(0,index), ya = x1_(1,index);
  double xb = x2_(0,index), yb = x2_(1,index);

  double a, b, c, d;
  // Transfer error in image 2
  if(side) *side=1;
  a = F(0,0) * xa + F(0,1) * ya + F(0,2);
  b = F(1,0) * xa + F(1,1) * ya + F(1,2);
  c = F(2,0) * xa + F(2,1) * ya + F(2,2);
  d = a*xb + b*yb + c;
  double err =  (d*d) / (a*a + b*b);
  // Transfer error in image 1
  if(symError_) { // ... but only if requested
    a = F(0,0) * xb + F(1,0) * yb + F(2,0);
    b = F(0,1) * xb + F(1,1) * yb + F(2,1);
    c = F(0,2) * xb + F(1,2) * yb + F(2,2);
    d = a*xa + b*ya + c;
    double err1 =  (d*d) / (a*a + b*b);
    if(err1>err) {
      err = err1;
      if(side) *side=0;
    }
  }
  return err;
}

}  // namespace orsa
