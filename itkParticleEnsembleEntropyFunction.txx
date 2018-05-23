/*=========================================================================
  Program:   ShapeWorks: Particle-based Shape Correspondence & Visualization
  Module:    $RCSfile: itkParticleEnsembleEntropyFunction.txx,v $
  Date:      $Date: 2011/03/24 01:17:33 $
  Version:   $Revision: 1.2 $
  Author:    $Author: wmartin $

  Copyright (c) 2009 Scientific Computing and Imaging Institute.
  See ShapeWorksLicense.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.
=========================================================================*/
#ifndef __itkParticleEnsembleEntropyFunction_txx
#define __itkParticleEnsembleEntropyFunction_txx

#include "itkParticleImageDomainWithGradients.h"
#include "vnl/algo/vnl_symmetric_eigensystem.h"
#include "itkParticleGaussianModeWriter.h"
#include <string>

namespace itk
{
template <unsigned int VDimension>
void
ParticleEnsembleEntropyFunction<VDimension>
::WriteModes(const std::string &prefix, int n) const
{
    typename ParticleGaussianModeWriter<VDimension>::Pointer writer =
            ParticleGaussianModeWriter<VDimension>::New();
    writer->SetShapeMatrix(m_ShapeMatrix);
    writer->SetFileName(prefix.c_str());
    writer->SetNumberOfModes(n);
    writer->Update();
}

template <unsigned int VDimension>
void
ParticleEnsembleEntropyFunction<VDimension>
::ComputeCovarianceMatrix()
{ 
    // NOTE: This code requires that indices be contiguous, i.e. it wont work if
    // you start deleting particles.
    const unsigned int num_samples = m_ShapeMatrix->cols();
    const unsigned int num_dims    = m_ShapeMatrix->rows();


    // Do we need to resize the covariance matrix?
    if (m_PointsUpdate.rows() != num_dims || m_PointsUpdate.cols() != num_samples)
    {
        m_PointsUpdate.set_size(num_dims, num_samples);
        m_PointsUpdate = m_PointsUpdate.fill(0.0);
    }
    vnl_matrix_type points_minus_mean;
    points_minus_mean.clear();
    points_minus_mean.set_size(num_dims, num_samples);
    points_minus_mean.fill(0.0);

    points_mean.clear();
    points_mean.set_size(num_dims, 1);

    // Compute the covariance matrix.
    // (A is D' in Davies paper)
    // Compute the mean shape vector.
    double _total = 0.0;
    for (unsigned int j = 0; j < num_dims; j++)
    {
        double total = 0.0;
        for (unsigned int i = 0; i < num_samples; i++)
        {
            total += m_ShapeMatrix->operator()(j, i);
        }
        points_mean(j,0) = total/(double)num_samples;
        _total += total;
    }


    for (unsigned int j = 0; j < num_dims; j++)
    {
        for (unsigned int i = 0; i < num_samples; i++)
        {
            points_minus_mean(j, i) = m_ShapeMatrix->operator()(j, i) - points_mean(j,0);
        }
    }
//    std:cout << points_minus_mean.extract(num_dims, num_samples, 0, 0) << std::endl;

#ifdef PARTICLE_DEBUG
    std::cout << "Shape Matrix : " << std::endl;
    std::cout << "total : " << _total << std::endl;
    for (unsigned int j = 0; j < num_dims; j++)
    {
        for(unsigned int i = 0; i < num_samples; i++)
        {
            std::cout << points_minus_mean(j,i) << " " ;
        }
        std::cout << "\n";
    }
    std::cout << "Done\n";
#endif

    vnl_diag_matrix<double> W;

    m_InverseCovMatrix.set_size(num_dims, num_dims);
    m_InverseCovMatrix.fill(0.0);
    vnl_matrix_type pinvMat(num_samples, num_samples, 0.0);

    if (this->m_UseMeanEnergy)
    {
        pinvMat.set_identity();
        m_InverseCovMatrix.set_identity();
    }
    else
    {
        vnl_svd <double> svd(points_minus_mean);

        vnl_matrix_type U = svd.U();
        vnl_matrix_type V = svd.V();

        W = svd.W();

        vnl_diag_matrix<double> invLambda = svd.W()*svd.W();

        invLambda.set_diagonal(invLambda.get_diagonal()/(double)(num_samples-1) + m_MinimumVariance);
        invLambda.invert_in_place();

        pinvMat = (V * invLambda) * V.transpose();
        m_InverseCovMatrix = (U * invLambda) * U.transpose();
    }
    m_PointsUpdate = points_minus_mean * pinvMat;

//     std::cout << m_PointsUpdate.extract(num_dims, num_samples,0,0) << std::endl;

    m_CurrentEnergy = 0.0;

    if (m_UseMeanEnergy)
        m_CurrentEnergy = points_minus_mean.frobenius_norm();
    else
    {
        m_MinimumEigenValue = W(0)*W(0) + m_MinimumVariance;
        for (unsigned int i = 0; i < num_samples; i++)
        {
            double val_i = W(i)*W(i) + m_MinimumVariance;
            if ( val_i < m_MinimumEigenValue)
                m_MinimumEigenValue = val_i;
            m_CurrentEnergy += log(val_i);
        }
    }

    m_CurrentEnergy /= 2.0;

    if (!m_UseMeanEnergy)
    {
        for (unsigned int i = 0; i < num_samples; i++)
        {
            std::cout << i << ": "<< W(i)*W(i) << std::endl;
        }
        std::cout << "Correspondence_ENERGY = " << m_CurrentEnergy << "\t MinimumVariance = "
                  << m_MinimumVariance <<  std::endl;
    }
    else
    {
        m_MinimumEigenValue = m_CurrentEnergy / 2.0;
        std::cout << "CorrMean_ENERGY = " << m_CurrentEnergy << std::endl;
    }
}

template <unsigned int VDimension>
typename ParticleEnsembleEntropyFunction<VDimension>::VectorType
ParticleEnsembleEntropyFunction<VDimension>
::Evaluate(unsigned int idx, unsigned int d, const ParticleSystemType * system,
           double &maxdt, double &energy) const
{
    // NOTE: This code requires that indices be contiguous, i.e. it won't work if
    // you start deleting particles.
    const unsigned int DomainsPerShape = m_ShapeMatrix->GetDomainsPerShape();

    maxdt  = m_MinimumEigenValue;

    VectorType gradE;
    unsigned int k = 0;
    int dom = d % DomainsPerShape;
    for (int i = 0; i < dom; i++)
        k += system->GetNumberOfParticles(i) * VDimension;
    k += idx*VDimension;

    vnl_matrix_type Xi(3,1,0.0);
    Xi(0,0) = m_ShapeMatrix->operator()(k  , d/DomainsPerShape) - points_mean(k, 0);
    Xi(1,0) = m_ShapeMatrix->operator()(k+1, d/DomainsPerShape) - points_mean(k+1, 0);
    Xi(2,0) = m_ShapeMatrix->operator()(k+2, d/DomainsPerShape) - points_mean(k+2, 0);


    vnl_matrix_type tmp1 = m_InverseCovMatrix.extract(3,3,k,k);

    vnl_matrix_type tmp = Xi.transpose()*tmp1;

    tmp *= Xi;

    energy = tmp(0,0);

    for (unsigned int i = 0; i< VDimension; i++)
    {
        gradE[i] = m_PointsUpdate(k + i, d / DomainsPerShape);
    }


    return system->TransformVector(gradE,
                                   system->GetInversePrefixTransform(d) *
                                   system->GetInverseTransform(d));
}

} // end namespace
#endif
