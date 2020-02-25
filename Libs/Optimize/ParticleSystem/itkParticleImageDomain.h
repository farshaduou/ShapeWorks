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
  void SetRegion(const PointType& lowerBound, const PointType& upperBound)
  {
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


    // Compute gradient image and set up gradient interpolation.
    typename GradientImageFilterType::Pointer filter = GradientImageFilterType::New();
    filter->SetInput(I);
    filter->SetUseImageSpacingOn();
    filter->Update();
    m_GradientImage = filter->GetOutput();

    m_GradientInterpolator->SetInputImage(m_GradientImage);
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
  inline VnlVectorType SampleNormalVnl(const PointType& p, T epsilon = 1.0e-5) const
  {
      VnlVectorType grad = this->SampleGradientVnl(p).normalize();
      return grad;
  }

  itkGetObjectMacro(GradientImage, GradientImageType);
  itkGetObjectMacro(Image, ImageType);
  itkGetConstObjectMacro(Image, ImageType);

  /** Sample the image at a point.  This method performs no bounds checking.
      To check bounds, use IsInsideBuffer. */
  inline T Sample(const PointType &p) const {
      if(IsInsideBuffer(p))
        return  m_ScalarInterpolator->Evaluate(p);
      else
        return 0.0;
  }

  /** Check whether the point p may be sampled in this image domain. */
  inline bool IsInsideBuffer(const PointType &p) const { 
      return m_ScalarInterpolator->IsInsideBuffer(p); 
  }

  /** Used when a domain is fixed. */
  void DeleteImages() {
    m_Image = 0;
    m_ScalarInterpolator = 0;
    m_GradientImage = 0;
    m_GradientInterpolator = 0;
  }

  /** Allow public access to the scalar interpolator. */
  itkGetObjectMacro(ScalarInterpolator, ScalarInterpolatorType);
  
protected:
  ParticleImageDomain() {
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
};

} // end namespace itk

#endif
