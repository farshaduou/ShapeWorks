/*=========================================================================
  Program:   ShapeWorks: Particle-based Shape Correspondence & Visualization
  Module:    $RCSfile: itkParticleImageDomain.h,v $
  Date:      $Date: 2011/03/24 01:17:33 $
  Version:   $Revision: 1.2 $
  Author:    $Author: wmartin $

  Copyright (c) 2009 Scientific Computing and Imaging Institute.
  See ShapeWorksLicense.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.
=========================================================================*/
#ifndef __itkParticleImageDomain_h
#define __itkParticleImageDomain_h

#include "itkImage.h"
#include "itkParticleDomain.h"
#include "itkLinearInterpolateImageFunction.h"

#include "itkVectorLinearInterpolateImageFunction.h"
#include "itkGradientImageFilter.h"
#include "itkFixedArray.h"

#include "itkDiscreteGaussianImageFilter.h"
#include "itkDerivativeImageFilter.h"

namespace itk
{
/** \class ParticleImageDomain
 *  A bounding-box region domain that sets its bounding box according to the
 *  origin, spacing, and RequestedRegion of a specified itk::Image.  This
 *  Domain object may be sampled for interpolated image values using the
 *  Sample(Point) method.
 *
 * \sa ParticleImageDomainWithGradients
 * \sa ParticleRegionDomain
 *
 */
template <class T, unsigned int VDimension=3>
class ITK_EXPORT ParticleImageDomain : public ParticleDomain<VDimension>
{
public:

  /** Type of the ITK image used by this class. */
  typedef Image<T, VDimension> ImageType;
  typedef GradientImageFilter<ImageType> GradientImageFilterType;
  typedef typename GradientImageFilterType::OutputImageType GradientImageType;
  typedef VectorLinearInterpolateImageFunction<GradientImageType, typename PointType::CoordRepType> GradientInterpolatorType;

  typedef FixedArray<T, 3> VectorType;
  typedef vnl_vector_fixed<T, 3> VnlVectorType;
  typedef vnl_matrix_fixed<T, VDimension, VDimension> VnlMatrixType;

  /** Point type of the domain (not the image). */
  typedef Point<double, VDimension> PointType;

  typedef LinearInterpolateImageFunction<ImageType, typename PointType::CoordRepType>
  ScalarInterpolatorType;

  /** Moves the point inside of the boundaries of the image */
  virtual bool ApplyConstraints(PointType& p) const {
      bool changed = false;
      for (unsigned int i = 0; i < VDimension; i++)
      {
          if (p[i] < GetLowerBound()[i]) {
              changed = true;
              p[i] = GetLowerBound()[i];
          }
          else if (p[i] > GetUpperBound()[i]) {
              changed = true;
              p[i] = GetUpperBound()[i];
          }
      }
      return changed;
  }
  /** This method is called by an optimizer after a call to Evaluate and may be
    used to apply any constraints the resulting vector, such as a projection
    to the surface tangent plane. Returns true if the gradient was modified.*/
  virtual bool ApplyVectorConstraints(
                                      vnl_vector_fixed<double, 
                                      VDimension>& gradE,
                                      const PointType& pos,
                                      double maxtimestep) const {
      if (this->m_ConstraintsEnabled == true) {
          const double epsilon = 1.0e-10;
          double dotprod = 0.0;
          VnlVectorType normal = this->SampleNormalVnl(pos, epsilon);
          for (unsigned int i = 0; i < VDimension; i++) { dotprod += normal[i] * gradE[i]; }
          for (unsigned int i = 0; i < VDimension; i++) { gradE[i] -= normal[i] * dotprod; }
          return true;
      }
      return false;
  }

  /** Set the lower/upper bound of the bounded region. */
  itkSetMacro(LowerBound, PointType);
  itkSetMacro(UpperBound, PointType);
  virtual const PointType& GetUpperBound() const { return m_UpperBound; }
  virtual const PointType& GetLowerBound() const { return m_LowerBound; }
  /** Specify the lower and upper bounds of the region. */
  void SetRegion(const PointType& lowerBound, const PointType& upperBound) {
      SetLowerBound(lowerBound);
      SetUpperBound(upperBound);
  }

  /** Set/Get the itk::Image specifying the particle domain.  The set method
      modifies the parent class LowerBound and UpperBound. */
  void SetImage(ImageType *I) {
    this->Modified();
    m_Image= I;

    // Set up the scalar image and interpolation.
    m_ScalarInterpolator->SetInputImage(m_Image);

    // Grab the upper-left and lower-right corners of the bounding box.  Points
    // are always in physical coordinates, not image index coordinates.
    typename ImageType::RegionType::IndexType idx
      = m_Image->GetRequestedRegion().GetIndex(); // upper lh corner
    typename ImageType::RegionType::SizeType sz
      = m_Image->GetRequestedRegion().GetSize();  // upper lh corner

    typename ImageType::PointType l0;
    m_Image->TransformIndexToPhysicalPoint(idx, l0);
    for (unsigned int i = 0; i < VDimension; i++)
        idx[i] += sz[i]-1;

    typename ImageType::PointType u0;
    m_Image->TransformIndexToPhysicalPoint(idx, u0);

    // Cast points to higher precision if needed.  Parent class uses doubles
    // because they are compared directly with points in the particle system,
    // which are always double precision.
    typename PointType l;
    typename PointType u;
    
    for (unsigned int i = 0; i < VDimension; i++)
      {
      l[i] = static_cast<double>(l0[i]);
      u[i] = static_cast<double>(u0[i]);
      }
    
    this->SetLowerBound(l);
    this->SetUpperBound(u);

    InitializeGradientImage();
    InitializeHessianImage();
  }

  /** Sample the image at a point.  This method performs no bounds checking.
      To check bounds, use IsInsideBuffer. */
  inline T Sample(const PointType& p) const {
      if (IsInsideBuffer(p))
          return  m_ScalarInterpolator->Evaluate(p);
      else
          return 0.0;
  }
  /** Sample the image at a point.  This method performs no bounds checking.
    To check bounds, use IsInsideBuffer.  SampleGradientsVnl returns a vnl
    vector of length VDimension instead of an itk::CovariantVector
    (itk::FixedArray). */
  inline VectorType SampleGradient(const PointType& p) const
  {
      if (this->IsInsideBuffer(p))
          return  m_GradientInterpolator->Evaluate(p);
      else {
          itkExceptionMacro("Gradient queried for a Point, " << p << ", outside the given image domain.");
          VectorType g(1.0e-5);
          return g;
      }
  }
  inline VnlVectorType SampleGradientVnl(const PointType& p) const {
      return VnlVectorType(this->SampleGradient(p).GetDataPointer());
  }
  inline VnlVectorType SampleNormalVnl(const PointType& p, T epsilon = 1.0e-5) const {
      VnlVectorType grad = this->SampleGradientVnl(p).normalize();
      return grad;
  }
  /** Sample the Hessian at a point.  This method performs no bounds checking.
    To check bounds, use IsInsideBuffer.  SampleHessiansVnl returns a vnl
    matrix of size VDimension x VDimension. */
  inline VnlMatrixType SampleHessianVnl(const PointType& p) const
  {
      VnlMatrixType ans;
      for (unsigned int i = 0; i < VDimension; i++)
      {
          ans[i][i] = m_Interpolators[i]->Evaluate(p);
      }

      // Cross derivatives
      unsigned int k = VDimension;
      for (unsigned int i = 0; i < VDimension; i++)
      {
          for (unsigned int j = i + 1; j < VDimension; j++, k++)
          {
              ans[i][j] = ans[j][i] = m_Interpolators[k]->Evaluate(p);
          }
      }
      return ans;
  }

  /** Allow public access to the scalar interpolator. */
  itkGetObjectMacro(ScalarInterpolator, ScalarInterpolatorType);
  itkGetObjectMacro(GradientImage, GradientImageType);
  itkGetObjectMacro(Image, ImageType);
  itkGetConstObjectMacro(Image, ImageType);
  /** Set /Get the standard deviation for blurring the image prior to
      computation of the Hessian derivatives.  This value must be set prior to
      initializing this class with an input image pointer and cannot be changed
      once the class is initialized.. */
  itkSetMacro(Sigma, double);
  itkGetMacro(Sigma, double);
  /** Access interpolators and partial derivative images. */
  typename ScalarInterpolatorType::Pointer* GetInterpolators() {
      return m_Interpolators;
  }
  typename ImageType::Pointer* GetPartialDerivatives() {
      return m_PartialDerivatives;
  }

  /** Check whether the point p may be sampled in this image domain. */
  inline bool IsInsideBuffer(const PointType &p) const { 
      return m_ScalarInterpolator->IsInsideBuffer(p); 
  }

  void DeletePartialDerivativeImages() {
      for (unsigned int i = 0; i < VDimension + ((VDimension * VDimension) - VDimension) / 2; i++) {
          m_PartialDerivatives[i] = 0;
          m_Interpolators[i] = 0;
      }
  }
  /** Used when a domain is fixed. */
  void DeleteImages() {
    m_Image = 0;
    m_ScalarInterpolator = 0;
    m_GradientImage = 0;
    m_GradientInterpolator = 0;
    this->DeletePartialDerivativeImages();
  }

  
protected:
  ParticleImageDomain() : m_Sigma(0.0) {
    m_ScalarInterpolator = ScalarInterpolatorType::New();
    m_GradientInterpolator = GradientInterpolatorType::New();
  }

  void PrintSelf(std::ostream& os, Indent indent) const {
    Superclass::PrintSelf(os, indent);

    os << indent << "LowerBound = " << GetLowerBound() << std::endl;
    os << indent << "UpperBound = " << GetUpperBound() << std::endl;
    os << indent << "m_Image = " << m_Image << std::endl;
    os << indent << "m_ScalarInterpolator = " << m_ScalarInterpolator << std::endl;
    os << indent << "m_GradientImage = " << m_GradientImage << std::endl;
    os << indent << "m_GradientInterpolator = " << m_GradientInterpolator << std::endl;
  }
  virtual ~ParticleImageDomain() {};
  
private:
  ParticleImageDomain(const ParticleImageDomain&); //purposely not implemented
  void operator=(const ParticleImageDomain&); //purposely not implemented

  typename ImageType::Pointer m_Image;
  typename ScalarInterpolatorType::Pointer m_ScalarInterpolator;

  PointType m_LowerBound;
  PointType m_UpperBound;

  typename GradientImageType::Pointer m_GradientImage;
  typename GradientInterpolatorType::Pointer m_GradientInterpolator;

  double m_Sigma;

  // Partials are stored:
  //     0: dxx  3: dxy  4: dxz
  //                 1: dyy  5: dyz
  //                            2: dzz
  //
  typename ImageType::Pointer  m_PartialDerivatives[VDimension + ((VDimension * VDimension) - VDimension) / 2];

  typename ScalarInterpolatorType::Pointer m_Interpolators[VDimension + ((VDimension * VDimension) - VDimension) / 2];

  void InitializeGradientImage() {
      // Compute gradient image and set up gradient interpolation.
      typename GradientImageFilterType::Pointer filter = GradientImageFilterType::New();
      filter->SetInput(this->GetImage());
      filter->SetUseImageSpacingOn();
      filter->Update();
      m_GradientImage = filter->GetOutput();

      m_GradientInterpolator->SetInputImage(m_GradientImage);
  }

  void InitializeHessianImage() {
      typename DiscreteGaussianImageFilter<ImageType, ImageType>::Pointer
          gaussian = DiscreteGaussianImageFilter<ImageType, ImageType>::New();
      gaussian->SetVariance(m_Sigma * m_Sigma);
      gaussian->SetInput(this->GetImage());
      gaussian->SetUseImageSpacingOn();
      gaussian->Update();

      // Compute the second derivatives and set up the interpolators
      for (unsigned int i = 0; i < VDimension; i++)
      {
          typename DerivativeImageFilter<ImageType, ImageType>::Pointer
              deriv = DerivativeImageFilter<ImageType, ImageType>::New();
          deriv->SetInput(gaussian->GetOutput());
          deriv->SetDirection(i);
          deriv->SetOrder(2);
          deriv->SetUseImageSpacingOn();
          deriv->Update();

          m_PartialDerivatives[i] = deriv->GetOutput();

          m_Interpolators[i] = ScalarInterpolatorType::New();
          m_Interpolators[i]->SetInputImage(m_PartialDerivatives[i]);
      }

      // Compute the cross derivatives and set up the interpolators
      unsigned int k = VDimension;
      for (unsigned int i = 0; i < VDimension; i++)
      {
          for (unsigned int j = i + 1; j < VDimension; j++, k++)
          {
              typename DerivativeImageFilter<ImageType, ImageType>::Pointer
                  deriv1 = DerivativeImageFilter<ImageType, ImageType>::New();
              deriv1->SetInput(gaussian->GetOutput());
              deriv1->SetDirection(i);
              deriv1->SetUseImageSpacingOn();
              deriv1->SetOrder(1);
              deriv1->Update();

              typename DerivativeImageFilter<ImageType, ImageType>::Pointer
                  deriv2 = DerivativeImageFilter<ImageType, ImageType>::New();
              deriv2->SetInput(deriv1->GetOutput());
              deriv2->SetDirection(j);
              deriv2->SetUseImageSpacingOn();
              deriv2->SetOrder(1);

              deriv2->Update();

              m_PartialDerivatives[k] = deriv2->GetOutput();
              m_Interpolators[k] = ScalarInterpolatorType::New();
              m_Interpolators[k]->SetInputImage(m_PartialDerivatives[k]);
          }
      }
  }
};

} // end namespace itk

#endif
