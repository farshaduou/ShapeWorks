#ifndef __PSMImageDomainWithGradients_h
#define __PSMImageDomainWithGradients_h

#include <itkImage.h>
#include <itkImageDuplicator.h>
#include "PSMImageDomain.h"
#include <itkVectorLinearInterpolateImageFunction.h>
#include <itkGradientImageFilter.h>
#include <itkFixedArray.h>

//using namespace itk;


/** \class PSMImageDomainWithGradients
 *
 * An image domain that extends PSMImageDomainWithGradients with image
 * gradient information.  Gradient values are interpolated with the
 * SampleGradients(point) method.
 *
 * \sa PSMImageDomain
 * \sa ParticleClipRegionDomain
 * \sa PSMDomain
 */
template <class T, unsigned int VDimension>
class PSMImageDomainWithGradients : public PSMImageDomain<T, VDimension>
{
public:
  /** Standard class typedefs */
  typedef PSMImageDomainWithGradients Self;
  typedef PSMImageDomain<T, VDimension> Superclass;
  typedef itk::SmartPointer<Self>  Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;
  typedef itk::WeakPointer<const Self>  ConstWeakPointer;

    /** Point type of the domain (not necessarily of the image). */
  typedef typename Superclass::PointType PointType;
  
  typedef typename Superclass::ImageType ImageType;
  typedef typename Superclass::ScalarInterpolatorType ScalarInterpolatorType;
  typedef itk::GradientImageFilter<ImageType> GradientImageFilterType;
  typedef typename GradientImageFilterType::OutputImageType GradientImageType;
  typedef itk::VectorLinearInterpolateImageFunction<GradientImageType, typename PointType::CoordRepType>
  GradientInterpolatorType;

  typedef itk::FixedArray<T, VDimension> VectorType;
  typedef vnl_vector_fixed<T, VDimension> VnlVectorType;
  
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(PSMImageDomainWithGradients, PSMImageDomain);

  /** Dimensionality of the domain of the particle system. */
  itkStaticConstMacro(Dimension, unsigned int, VDimension);

  /** Set/Get the itk::Image specifying the particle domain.  The set method
      modifies the parent class LowerBound and UpperBound. */
  void SetImage(ImageType *I)
  {
    Superclass::SetImage(I);
 
    // Compute gradient image and set up gradient interpolation.
    typename GradientImageFilterType::Pointer filter = GradientImageFilterType::New();
    filter->SetInput(I);
    filter->SetUseImageSpacingOn();
    filter->Update();
    m_GradientImage = filter->GetOutput();
    
    m_GradientInterpolator->SetInputImage(m_GradientImage);
  }
  itkGetObjectMacro(GradientImage, GradientImageType);


  /** Sample the image at a point.  This method performs no bounds checking.
      To check bounds, use IsInsideBuffer.  SampleGradientsVnl returns a vnl
      vector of length VDimension instead of an itk::CovariantVector
      (itk::FixedArray). */
  inline VectorType SampleGradient(const PointType &p) const
  {   return  m_GradientInterpolator->Evaluate(p);  }
  inline VnlVectorType SampleGradientVnl(const PointType &p) const
  { return VnlVectorType( this->SampleGradient(p).GetDataPointer() ); }
  inline VnlVectorType SampleNormalVnl(const PointType &p, T epsilon = 1.0e-5) const
  {
    VnlVectorType grad = this->SampleGradientVnl(p);
    double q = 1.0 / (grad.magnitude() + epsilon);
    for (unsigned int i = 0; i < VDimension; i++) { grad[i] *= q; }
    return grad;
  }
  
  /** Allow public access to the scalar interpolator. */
  itkGetObjectMacro(GradientInterpolator, GradientInterpolatorType);


  /** This method is called by an optimizer after a call to Evaluate and may be
      used to apply any constraints the resulting vector, such as a projection
      to the surface tangent plane. Returns true if the gradient was modified.*/
  virtual bool ApplyVectorConstraints(vnl_vector_fixed<double, VDimension> &gradE,
                                      const PointType &pos,
                                      double& maxtimestep) const
    
  {
    if (this->m_ConstraintsEnabled == true)
      {
      const double epsilon = 1.0e-10;
      
      double dotprod = 0.0;  
      VnlVectorType normal =  this->SampleNormalVnl(pos, epsilon);
      for (unsigned int i = 0; i < VDimension; i++) {   dotprod  += normal[i] * gradE[i]; }
      for (unsigned int i = 0; i < VDimension; i++) {   gradE[i] -= normal[i] * dotprod; }
     
      return true;
      }
    else return false;
  }
  
protected:
  PSMImageDomainWithGradients()
  {
    m_GradientInterpolator = GradientInterpolatorType::New();
  }

  void PrintSelf(std::ostream& os, itk::Indent indent) const
  {
    Superclass::PrintSelf(os, indent);
    os << indent << "m_GradientImage = " << m_GradientImage << std::endl;
    os << indent << "m_GradientInterpolator = " << m_GradientInterpolator << std::endl;
  }
  virtual ~PSMImageDomainWithGradients() {};
  
private:
  PSMImageDomainWithGradients(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typename GradientImageType::Pointer m_GradientImage;
  typename GradientInterpolatorType::Pointer m_GradientInterpolator;
};



#endif
