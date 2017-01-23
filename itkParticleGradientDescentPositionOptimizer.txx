/*=========================================================================
  Program:   ShapeWorks: Particle-based Shape Correspondence & Visualization
  Module:    $RCSfile: itkParticleGradientDescentPositionOptimizer.txx,v $
  Date:      $Date: 2011/03/24 01:17:33 $
  Version:   $Revision: 1.2 $
  Author:    $Author: wmartin $

  Copyright (c) 2009 Scientific Computing and Imaging Institute.
  See ShapeWorksLicense.txt for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.
=========================================================================*/
#ifndef __itkParticleGradientDescentPositionOptimizer_txx
#define __itkParticleGradientDescentPositionOptimizer_txx

#ifdef SW_USE_OPENMP
#include <omp.h>
const int global_iteration = 1;
#else /* SW_USE_OPENMP */
const int global_iteration = 1;
#endif /* SW_USE_OPENMP */

#include <algorithm>
#include <ctime>
#include <string>
#include "itkParticleImageDomainWithGradients.h"
#include <vector>
#include <fstream>
#include <sstream>
namespace itk
{
template <class TGradientNumericType, unsigned int VDimension>
ParticleGradientDescentPositionOptimizer<TGradientNumericType, VDimension>
::ParticleGradientDescentPositionOptimizer()
{
  m_StopOptimization = false;
  m_NumberOfIterations = 0;
  m_MaximumNumberOfIterations = 0;
  m_Tolerance = 0.0;
  m_TimeStep = 1.0;
  m_OptimizationMode = 0;
}

/*** ADAPTIVE GAUSS SEIDEL ***/
template <class TGradientNumericType, unsigned int VDimension>
void
ParticleGradientDescentPositionOptimizer<TGradientNumericType, VDimension>
::StartAdaptiveGaussSeidelOptimization()
{
    std::cout << "Starting Adaptive Gauss Seidel Optimization ..... \n" << std::flush;
  const double factor = 1.1;//1.1;
  //  const double epsilon = 1.0e-4;

  // NOTE: THIS METHOD WILL NOT WORK AS WRITTEN IF PARTICLES ARE
  // ADDED TO THE SYSTEM DURING OPTIMIZATION.
  m_StopOptimization = false;
  m_GradientFunction->SetParticleSystem(m_ParticleSystem);

  typedef typename DomainType::VnlVectorType NormalType;
  std::vector<PointType> meanPositions;
  std::vector<NormalType> meanNormals;

  bool reset = false;
  // Make sure the time step vector is the right size
  while (m_TimeSteps.size() != m_ParticleSystem->GetNumberOfDomains() )
    {
    reset = true;
    std::vector<double> tmp;
    m_TimeSteps.push_back( tmp );
    }

  for (unsigned int i = 0; i < m_ParticleSystem->GetNumberOfDomains(); i++)
    {
    unsigned int np = m_ParticleSystem->GetPositions(i)->GetSize();
    if (m_TimeSteps[i].size() != np)
      {
      // resize and initialize everything to 1.0
      m_TimeSteps[i].resize(np);
      for (unsigned int j = 0; j < np; j++)
        {
        m_TimeSteps[i][j] = 1.0;
        }
      reset = true;
      }
    }

  //  if (reset == true) m_GradientFunction->ResetBuffers();
  
  const double pi = std::acos(-1.0);
  unsigned int numdomains = m_ParticleSystem->GetNumberOfDomains();
  std::vector<double> meantime(numdomains);
  std::vector<double> maxtime(numdomains);
  std::vector<double> mintime(numdomains);

  //Praful - randomize the order of updates on shapes
  std::srand(1);
  std::vector<int> randDomIndx(numdomains);
  for (int i = 0; i<numdomains; i++) randDomIndx[i]=i;

  unsigned int counter = 0;

  for (unsigned int q = 0; q < numdomains; q++)
    {
    meantime[q] = 0.0;
    maxtime[q]  = 1.0e30;
    mintime[q]  = 1.0;
    }

  double maxchange = 0.0;
  while (m_StopOptimization == false)
    {
      if (counter % global_iteration == 0)
      {
        std::cerr << "Performing global step\n" ;
        m_GradientFunction->BeforeIteration();
      }
      counter++;
      //Praful - randomize the order of updates on shapes
        if (m_randomOrdering == true)
            std::random_shuffle(randDomIndx.begin(), randDomIndx.end());

        if (m_debug_projection)
        {
            meanNormals.clear();
            meanPositions.clear();
            for (unsigned int j = 0; j < m_ParticleSystem->GetNumberOfParticles(); j++)
            {
                PointType meanPt;
                meanPt[0] = 0.0; meanPt[1] = 0.0; meanPt[2] = 0.0;
                NormalType meanNormal;
                meanNormal.fill(0.0);
                for (unsigned int dom = 0; dom < numdomains; dom++)
                {
                    const DomainType * domain = static_cast<const DomainType *>(m_ParticleSystem->GetDomain(dom));
                    PointType pt = m_ParticleSystem->GetPosition(j, dom);
                    NormalType normalPs = domain->SampleNormalVnl(pt);
                    for (int vd = 0; vd < VDimension; vd++)
                    {
                        meanPt[vd] += pt[vd]/numdomains;
                        meanNormal[vd] += normalPs[vd]/numdomains;
                    }
                }
                meanNormal.normalize();
                meanPositions.push_back(meanPt);
                meanNormals.push_back(meanNormal);
            }
        }

      //Praful
#pragma omp parallel
{

    // Iterate over each domain
#pragma omp for
//    for (int dom = 0; dom < numdomains; dom++)
//      {
          //Praful - randomize the order of updates on shapes
    for (int domId = 0; domId < numdomains; domId++)
      {
                int dom = randDomIndx[domId];
                //Praful
//                std::cout<<"==================================="<<std::endl;
//                std::cout<<domId << " " << dom << std::endl;
//                std::cout<<"==================================="<<std::endl;

                int tid = 1;
				int num_threads = 1;
				
#ifdef SW_USE_OPENMP
				tid = omp_get_thread_num() + 1;
				num_threads = omp_get_num_threads();
#endif /* SW_USE_OPENMP */


        //std::cerr << "[thread " << tid << "/" << num_threads << "] iterating on domain " << dom << "\n";
      meantime[dom] = 0.0;
      // skip any flagged domains
      if (m_ParticleSystem->GetDomainFlag(dom) == false)
        {
          double maxdt;

          VectorType gradient;
          VectorType original_gradient;
          PointType newpoint;

          const DomainType * domain = static_cast<const DomainType *>(m_ParticleSystem->GetDomain(dom));

          typename GradientFunctionType::Pointer localGradientFunction;

          localGradientFunction = m_GradientFunction;

#ifdef SW_USE_OPENMP
          localGradientFunction = m_GradientFunction->Clone();
#endif /* SW_USE_OPENMP */


        // Tell function which domain we are working on.
        localGradientFunction->SetDomainNumber(dom);
     
        // Iterate over each particle position
        unsigned int k = 0;
        maxchange = 0.0;
        typename ParticleSystemType::PointContainerType::ConstIterator endit =
          m_ParticleSystem->GetPositions(dom)->GetEnd();
        for (typename ParticleSystemType::PointContainerType::ConstIterator it
               = m_ParticleSystem->GetPositions(dom)->GetBegin(); it != endit; it++, k++)
          {
            //if((k+1) == 67)
              //  int test =0;
          bool done = false;
//          int iter = 0;

          // Compute gradient update.
          double energy = 0.0;
          localGradientFunction->BeforeEvaluate(it.GetIndex(), dom, m_ParticleSystem);
          original_gradient = localGradientFunction->Evaluate(it.GetIndex(), dom, m_ParticleSystem,
                                                           maxdt, energy);
          unsigned int idx = it.GetIndex();
          PointType pt = *it;
          NormalType ptNormalOld = domain->SampleNormalVnl(pt);
          if (m_debug_projection)
          {
              double dptOld = std::acos(dot_product(ptNormalOld, meanNormals[idx]));
              PointType diffPt;
              diffPt[0] = pt[0] - meanPositions[idx][0];
              diffPt[1] = pt[1] - meanPositions[idx][1];
              diffPt[2] = pt[2] - meanPositions[idx][2];
              double dstOld = std::sqrt(diffPt[0]*diffPt[0] + diffPt[1]*diffPt[1] + diffPt[2]*diffPt[2]);
          }

          double newenergy, gradmag;
          while ( !done )
            {
            gradient = original_gradient * m_TimeSteps[dom][k];

//            std::cerr << "Vec const -- Begin " << iter << "\n";
            dynamic_cast<DomainType *>(m_ParticleSystem->GetDomain(dom))
              ->ApplyVectorConstraints(gradient,
                                       m_ParticleSystem->GetPosition(it.GetIndex(), dom), maxdt);
            gradmag = gradient.magnitude();
//            std::cerr << "Vec const -- End " << iter << ' ' << gradmag << "\n";
//            iter++;
            
            // Prevent a move which is too large
            if (gradmag * m_TimeSteps[dom][k] > maxdt)
              {
              m_TimeSteps[dom][k] /= factor;
              }
            else // Move is not too large
              {
              // Make a move and compute new energy
              for (unsigned int i = 0; i < VDimension; i++)
                {  newpoint[i] = pt[i] - gradient[i]; }
              dynamic_cast<DomainType *>(m_ParticleSystem->GetDomain(dom))->ApplyConstraints(newpoint);
              m_ParticleSystem->SetPosition(newpoint, it.GetIndex(), dom);

              newenergy = localGradientFunction->Energy(it.GetIndex(), dom, m_ParticleSystem);
              
              if (newenergy < energy) // good move, increase timestep for next time
                {
                meantime[dom] += m_TimeSteps[dom][k];
                m_TimeSteps[dom][k] *= factor;
                if (m_TimeSteps[dom][k] > maxtime[dom]) m_TimeSteps[dom][k] = maxtime[dom];
                if (gradmag > maxchange) maxchange = gradmag;
                done = true;

                // SHIREEN: for debugging
                //double old_new_dist = sqrt(pow(pt[0] - newpoint[0], 2.0) + pow(pt[1] - newpoint[1], 2.0) + pow(pt[2] - newpoint[2], 2.0));
                //std::cout << pt[0] << ", " << pt[1] << ", " << pt[2] << ", " << newpoint[0] << ", " << newpoint[1] << ", " <<  newpoint[2] << ", " << old_new_dist << ",\n " << std::flush;
                }
              else 
                {// bad move, reset point position and back off on timestep            
                if (m_TimeSteps[dom][k] > mintime[dom])
                  {
                  dynamic_cast<DomainType *>(m_ParticleSystem->GetDomain(dom))->ApplyConstraints(pt);
                  m_ParticleSystem->SetPosition(pt, it.GetIndex(), dom);

                  m_TimeSteps[dom][k] /= factor;
                  }
                else // keep the move with timestep 1.0 anyway
                  {
//                  meantime[dom] += m_TimeSteps[dom][k];
//                  m_TimeSteps[dom][k] = mintime[dom];
//                  if (gradmag > maxchange) maxchange = gradmag;
                  done = true;
                  //std::cout << "\t A bad move will be kept at time step = " << m_TimeSteps[dom][k]*factor << std::endl << std::flush;

                  // SHIREEN: for debugging
                  //double old_new_dist = sqrt(pow(pt[0] - newpoint[0], 2.0) + pow(pt[1] - newpoint[1], 2.0) + pow(pt[2] - newpoint[2], 2.0));
                  //std::cout << pt[0] << ", " << pt[1] << ", " << pt[2] << ", " << newpoint[0] << ", " << newpoint[1] << ", " <<  newpoint[2] << ", " << old_new_dist << ",\n " << std::flush;
                  }
                }
              }
            } // end while not done

          if (m_debug_projection)
          {
              PointType ptNew = m_ParticleSystem->GetPosition(idx, dom);
              NormalType ptNormalNew = domain->SampleNormalVnl(ptNew);
              double dptNew = std::acos(dot_product(ptNormalNew, meanNormals[idx]));
              PointType diffPtNew;
              diffPtNew[0] = ptNew[0] - meanPositions[idx][0];
              diffPtNew[1] = ptNew[1] - meanPositions[idx][1];
              diffPtNew[2] = ptNew[2] - meanPositions[idx][2];
              double dstNew = std::sqrt(diffPtNew[0]*diffPtNew[0] + diffPtNew[1]*diffPtNew[1] + diffPtNew[2]*diffPtNew[2]);

              double devOldToNew = std::acos(dot_product(ptNormalNew, ptNormalOld));
              if (devOldToNew > pi/4.0)
                  std::cout << "Warning1!!! Iter# " << m_NumberOfIterations << " domain # " << dom <<" point # "<< idx << " normal turning angle = " << devOldToNew*180.0/pi << std::endl;
              //          if (dstNew > dstOld)
              //              std::cout << "Warning2!!! Iter# " << m_NumberOfIterations << " domain # " << dom <<" point # "<< idx << " distance to mean increasing... newDist = " << dstNew << " oldDist = " << dstOld <<std::endl;
              //          if (dptNew > dptOld)
              //              std::cout << "Warning3!!! Iter# " << m_NumberOfIterations << " domain # " << dom <<" point # "<< idx << " angle to mean increasing... newAngle = " << dptNew*180.0/pi << " oldAngle = " << dptOld*180.0/pi << std::endl;
          }



          } // for each particle
        
        // Compute mean time step
        meantime[dom] /= static_cast<double>(k);

        if (meantime[dom] < 1.0) meantime[dom] = 1.0;
        //        std::cout << "meantime = " << meantime[dom] << std::endl;
        maxtime[dom] = meantime[dom] + meantime[dom] * 0.2;
        mintime[dom] = meantime[dom] - meantime[dom] * 0.1;
        } // if not flagged
      }// for each domain
}

        //Praful - Newton Ralphson debugging
        //m_ParticleSystem->PrintDebugData(m_NumberOfIterations);

    m_NumberOfIterations++;
    m_GradientFunction->AfterIteration();
    this->InvokeEvent(itk::IterationEvent());
    
    // Check for convergence.  Optimization is considered to have converged if
    // max number of iterations is reached or maximum distance moved by any
    // particle is less than the specified precision.
    //    std::cout << "maxchange = " << maxchange << std::endl;
    if ((m_NumberOfIterations == m_MaximumNumberOfIterations)
      || (m_Tolerance > 0.0 &&  maxchange <  m_Tolerance) )
      {
      m_StopOptimization = true;
      }
    
    } // end while stop optimization  
}

/*** GAUSS SEIDEL ***/
template <class TGradientNumericType, unsigned int VDimension>
void
ParticleGradientDescentPositionOptimizer<TGradientNumericType, VDimension>
::StartGaussSeidelOptimization()
{
    std::cout << "Starting Gauss Seidel Optimization ..... \n" << std::flush;
  // NOTE: THIS METHOD WILL NOT WORK AS WRITTEN IF PARTICLES ARE
  // ADDED TO THE SYSTEM DURING OPTIMIZATION.
  m_StopOptimization = false;
  m_GradientFunction->SetParticleSystem(m_ParticleSystem);
  
  VectorType gradient;
  PointType newpoint;
  
  while (m_StopOptimization == false)
    {
    m_GradientFunction->BeforeIteration();
    double maxdt;
    
    // Iterate over each domain
    for (unsigned int dom = 0; dom < m_ParticleSystem->GetNumberOfDomains(); dom++)
      {
      // skip any flagged domains
      if (m_ParticleSystem->GetDomainFlag(dom) == false)
        {
        // Tell function which domain we are working on.
        m_GradientFunction->SetDomainNumber(dom);
        
        // Iterate over each particle position
        unsigned int k = 0;
        typename ParticleSystemType::PointContainerType::ConstIterator endit =
          m_ParticleSystem->GetPositions(dom)->GetEnd();
        for (typename ParticleSystemType::PointContainerType::ConstIterator it
               = m_ParticleSystem->GetPositions(dom)->GetBegin(); it != endit; it++, k++)
          {
          // Compute gradient update.
          m_GradientFunction->BeforeEvaluate(it.GetIndex(), dom, m_ParticleSystem);
          gradient = m_GradientFunction->Evaluate(it.GetIndex(), dom, m_ParticleSystem,
                                                  maxdt);
          
          // May modify gradient
          dynamic_cast<DomainType *>(m_ParticleSystem->GetDomain(dom))
            ->ApplyVectorConstraints(gradient,
                                     m_ParticleSystem->GetPosition(it.GetIndex(), dom), maxdt);
          
          // Hack to avoid blowing up under certain conditions.
          if (gradient.magnitude() > maxdt)
            { gradient = (gradient / gradient.magnitude()) * maxdt; }
          
          // Compute particle move based on update.
           for (unsigned int i = 0; i < VDimension; i++)
             {
             newpoint[i] = (*it)[i] - gradient[i] * m_TimeStep;
             }
           
           // Apply update
           dynamic_cast<DomainType *>(m_ParticleSystem->GetDomain(dom))->ApplyConstraints(newpoint);
           m_ParticleSystem->SetPosition(newpoint, it.GetIndex(), dom);
           } // for each particle
         } // if not flagged
       }// for each domain

     m_NumberOfIterations++;
     m_GradientFunction->AfterIteration();
     this->InvokeEvent(itk::IterationEvent());
  
     // Check for convergence.  Optimization is considered to have converged if
     // max number of iterations is reached or maximum distance moved by any
     // particle is less than the specified precision.
     if (m_NumberOfIterations == m_MaximumNumberOfIterations)
       {
       m_StopOptimization = true;
       }
  
     } // end while

 }

template <class TGradientNumericType, unsigned int VDimension>
void
ParticleGradientDescentPositionOptimizer<TGradientNumericType, VDimension>
::StartJacobiOptimization()
{
    std::cout << "Starting Jacobi Optimization ..... \n" << std::flush;

  // NOTE: THIS METHOD WILL NOT WORK AS WRITTEN IF PARTICLES ARE
  // ADDED TO THE SYSTEM DURING OPTIMIZATION.
  m_StopOptimization = false;

  m_GradientFunction->SetParticleSystem(m_ParticleSystem);
  
  // Vector for storing updates.
  std::vector<std::vector<PointType> > updates(m_ParticleSystem->GetNumberOfDomains());

  for (unsigned int d = 0; d < m_ParticleSystem->GetNumberOfDomains(); d++)
    {
    updates[d].resize(m_ParticleSystem->GetPositions(d)->GetSize());
    }
  VectorType gradient;
  PointType  newpoint;

  while (m_StopOptimization == false)
    {
    m_GradientFunction->BeforeIteration();
    double maxdt;
    
    // Iterate over each domain
    for (unsigned int dom = 0; dom < m_ParticleSystem->GetNumberOfDomains(); dom++)
      {
      // skip any flagged domains
      if (m_ParticleSystem->GetDomainFlag(dom) == false)
        {
        // Tell function which domain we are working on.
        m_GradientFunction->SetDomainNumber(dom);
        
        // Iterate over each particle position
        unsigned int k = 0;
        typename ParticleSystemType::PointContainerType::ConstIterator endit =
          m_ParticleSystem->GetPositions(dom)->GetEnd();
        for (typename ParticleSystemType::PointContainerType::ConstIterator it
               = m_ParticleSystem->GetPositions(dom)->GetBegin(); it != endit; it++, k++)
          {
          // Compute gradient update.
          m_GradientFunction->BeforeEvaluate(it.GetIndex(), dom, m_ParticleSystem);
          gradient = m_GradientFunction->Evaluate(it.GetIndex(), dom, m_ParticleSystem,
                                                  maxdt);
          // May modify gradient
          dynamic_cast<DomainType *>(m_ParticleSystem->GetDomain(dom))
            ->ApplyVectorConstraints(gradient, m_ParticleSystem->GetPosition(it.GetIndex(), dom), maxdt);
          
          // Hack to avoid blowing up under certain conditions.
          if (gradient.magnitude() > maxdt)
            {
            gradient = (gradient / gradient.magnitude()) * maxdt;
            }
          
          // Compute particle move based on update.
          for (unsigned int i = 0; i < VDimension; i++)
            {
            newpoint[i] = (*it)[i] - gradient[i] * m_TimeStep;
            }
          
          // Store update
          updates[dom][k] = newpoint;
          } // for each particle
        } // if not flagged
      }// for each domain
    
    for (unsigned int dom = 0; dom < m_ParticleSystem->GetNumberOfDomains(); dom++)
      {
      // skip any flagged domains
      if (m_ParticleSystem->GetDomainFlag(dom) == false)
        {
        unsigned int k = 0;
        typename ParticleSystemType::PointContainerType::ConstIterator endit =
          m_ParticleSystem->GetPositions(dom)->GetEnd();
        for (typename ParticleSystemType::PointContainerType::ConstIterator it
               = m_ParticleSystem->GetPositions(dom)->GetBegin();  it != endit; it++, k++)
          {
          dynamic_cast<DomainType *>(m_ParticleSystem->GetDomain(dom))->ApplyConstraints(updates[dom][k]);
          m_ParticleSystem->SetPosition(updates[dom][k], it.GetIndex(), dom);
          } // for each particle
        } // if not flagged
      } // for each domain
    
    m_NumberOfIterations++;
    m_GradientFunction->AfterIteration();
    this->InvokeEvent(itk::IterationEvent());
    
    // Check for convergence.  Optimization is considered to have converged if
    // max number of iterations is reached or maximum distance moved by any
    // particle is less than the specified precision.
    if (m_NumberOfIterations == m_MaximumNumberOfIterations)
      {
      m_StopOptimization = true;
      }
    
    } // end while
  
}

} // end namespace

#endif
