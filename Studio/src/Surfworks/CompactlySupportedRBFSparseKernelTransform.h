#ifndef _CompactlySupportedRBFSparseKernelTransform_h
#define _CompactlySupportedRBFSparseKernelTransform_h

#include "SparseKernelTransform.h"


/** \class CompactlySupportedRBFSparseKernelTransform
 * \ingroup Transforms
 */
template <class TScalarType,         // Data type for scalars (float or double)
          unsigned int NDimensions = 3>          // Number of dimensions
class CompactlySupportedRBFSparseKernelTransform :
        public SparseKernelTransform<   TScalarType, NDimensions>
{
public:
    /** Standard class typedefs. */
    typedef CompactlySupportedRBFSparseKernelTransform Self;
    typedef SparseKernelTransform<    TScalarType, NDimensions>   Superclass;
    typedef itk::SmartPointer<Self>        Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;

    /** New macro for creation of through a Smart Pointer */
    itkNewMacro( Self );

    /** Run-time type information (and related methods). */
    itkTypeMacro( CompactlySupportedRBFSparseKernelTransform, SparseKernelTransform );

    /** Scalar type. */
    typedef typename Superclass::ScalarType  ScalarType;

    /** Parameters type. */
    typedef typename Superclass::ParametersType  ParametersType;

    /** Jacobian Type */
    typedef typename Superclass::JacobianType  JacobianType;

    /** Dimension of the domain space. */
    itkStaticConstMacro(SpaceDimension, unsigned int,Superclass::SpaceDimension);

    /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited */
    typedef typename Superclass::InputPointType  InputPointType;
    typedef typename Superclass::OutputPointType  OutputPointType;
    typedef typename Superclass::InputVectorType InputVectorType;
    typedef typename Superclass::OutputVectorType OutputVectorType;
    typedef typename Superclass::InputCovariantVectorType InputCovariantVectorType;
    typedef typename Superclass::OutputCovariantVectorType OutputCovariantVectorType;
    typedef typename Superclass::PointsIterator PointsIterator;
    //  void SetParameters( const ParametersType & parameters );

    void SetSigma(double sigma){this->Sigma = sigma;}

    virtual void ComputeJacobianWithRespectToParameters(
        const InputPointType  &in, JacobianType &jacobian) const;


protected:
    CompactlySupportedRBFSparseKernelTransform() {this->Sigma = 1; }
    virtual ~CompactlySupportedRBFSparseKernelTransform() {}

    /** These (rather redundant) typedefs are needed because on SGI, typedefs
   * are not inherited. */
    typedef typename Superclass::GMatrixType GMatrixType;

    const GMatrixType & ComputeG(const InputVectorType & x) const;

    /** Compute the contribution of the landmarks weighted by the kernel funcion
      to the global deformation of the space  */
    virtual void ComputeDeformationContribution( const InputPointType & inputPoint,
                                                 OutputPointType & result ) const;

private:
    CompactlySupportedRBFSparseKernelTransform(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    // basis support
    double Sigma;

};




#ifndef ITK_MANUAL_INSTANTIATION
#include "CompactlySupportedRBFSparseKernelTransform.hxx"
#endif

#endif // _CompactlySupportedRBFSparseKernelTransform_h
