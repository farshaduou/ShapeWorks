#ifndef __ThinPlateSplineKernelTransform2_h
#define __ThinPlateSplineKernelTransform2_h

#include "KernelTransform2.h"

//using namespace itk;

/** \class ThinPlateSplineKernelTransform2
 * This class defines the thin plate spline (TPS) transformation.
 * It is implemented in as straightforward a manner as possible from
 * the IEEE TMI paper by Davis, Khotanzad, Flamig, and Harms,
 * Vol. 16 No. 3 June 1997
 *
 * \ingroup Transforms
 */
template< class TScalarType,         // Data type for scalars (float or double)
unsigned int NDimensions = 3 >
// Number of dimensions
class ThinPlateSplineKernelTransform2 :
  public KernelTransform2< TScalarType, NDimensions >
{
public:

  /** Standard class typedefs. */
  typedef ThinPlateSplineKernelTransform2              Self;
  typedef KernelTransform2< TScalarType, NDimensions > Superclass;
  typedef itk::SmartPointer< Self >                         Pointer;
  typedef itk::SmartPointer< const Self >                   ConstPointer;

  /** New macro for creation of through a Smart Pointer */
  itkNewMacro( Self );

  /** Run-time type information (and related methods). */
  itkTypeMacro( ThinPlateSplineKernelTransform2, KernelTransform2 );

  /** Scalar type. */
  typedef typename Superclass::ScalarType ScalarType;

  /** Parameters type. */
  typedef typename Superclass::ParametersType ParametersType;

  /** Jacobian Type */
  typedef typename Superclass::JacobianType JacobianType;

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, Superclass::SpaceDimension );

  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited.
   */
  typedef typename Superclass::InputPointType            InputPointType;
  typedef typename Superclass::OutputPointType           OutputPointType;
  typedef typename Superclass::InputVectorType           InputVectorType;
  typedef typename Superclass::OutputVectorType          OutputVectorType;
  typedef typename Superclass::InputCovariantVectorType  InputCovariantVectorType;
  typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
  typedef typename Superclass::PointsIterator            PointsIterator;

  void SetSigma(double sigma){}; // this is only to match the compact supported class

protected:

  ThinPlateSplineKernelTransform2()
  {
    this->m_FastComputationPossible = true;
  }


  virtual ~ThinPlateSplineKernelTransform2() {}

  /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited.
   */
  typedef typename Superclass::GMatrixType GMatrixType;

  /** Compute G(x)
   * For the thin plate spline, this is:
   * G(x) = r(x)*I
   * \f$ G(x) = r(x)*I \f$
   * where
   * r(x) = Euclidean norm = sqrt[x1^2 + x2^2 + x3^2]
   * \f[ r(x) = \sqrt{ x_1^2 + x_2^2 + x_3^2 }  \f]
   * I = identity matrix.
   */
  void ComputeG( const InputVectorType & x, GMatrixType & GMatrix ) const;

  /** Compute the contribution of the landmarks weighted by the kernel function
   * to the global deformation of the space.
   */
  virtual void ComputeDeformationContribution(
    const InputPointType & inputPoint, OutputPointType & result ) const;

private:

  ThinPlateSplineKernelTransform2( const Self & ); // purposely not implemented
  void operator=( const Self & );                  // purposely not implemented

};

#include "ThinPlateSplineKernelTransform2.cpp"

#endif // __ThinPlateSplineKernelTransform2_h
