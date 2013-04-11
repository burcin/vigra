/************************************************************************/
/*                                                                      */
/*               Copyright 2011-2013 by Ullrich Koethe                  */
/*                                                                      */
/*    This file is part of the VIGRA computer vision library.           */
/*    The VIGRA Website is                                              */
/*        http://hci.iwr.uni-heidelberg.de/vigra/                       */
/*    Please direct questions, bug reports, and contributions to        */
/*        ullrich.koethe@iwr.uni-heidelberg.de    or                    */
/*        vigra@informatik.uni-hamburg.de                               */
/*                                                                      */
/*    Permission is hereby granted, free of charge, to any person       */
/*    obtaining a copy of this software and associated documentation    */
/*    files (the "Software"), to deal in the Software without           */
/*    restriction, including without limitation the rights to use,      */
/*    copy, modify, merge, publish, distribute, sublicense, and/or      */
/*    sell copies of the Software, and to permit persons to whom the    */
/*    Software is furnished to do so, subject to the following          */
/*    conditions:                                                       */
/*                                                                      */
/*    The above copyright notice and this permission notice shall be    */
/*    included in all copies or substantial portions of the             */
/*    Software.                                                         */
/*                                                                      */
/*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND    */
/*    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES   */
/*    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND          */
/*    NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT       */
/*    HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,      */
/*    WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING      */
/*    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR     */
/*    OTHER DEALINGS IN THE SOFTWARE.                                   */
/*                                                                      */
/* Most of EMDComputerRubner class was imported from source code
 * available at
 *
 *      http://robotics.stanford.edu/~rubner/emd/default.htm
 *
 * Copyright statement as follows:
 *
 *  emd.h
 *
 *  Last update: 3/24/98
 *
 *  An implementation of the Earth Movers Distance.
 *  Based of the solution for the Transportation problem as described in
 *  "Introduction to Mathematical Programming" by F. S. Hillier and
 *  G. J. Lieberman, McGraw-Hill, 1990.
 *
 *  Copyright (C) 1998 Yossi Rubner
 *  Computer Science Department, Stanford University
 *  E-Mail: rubner@cs.stanford.edu   URL: http://vision.stanford.edu/~rubner
 ***********************************************************************/

#ifndef VIGRA_EMD_HXX
#define VIGRA_EMD_HXX

#include <vigra/error.hxx>
#include <vigra/array_vector.hxx>
#include <vigra/matrix.hxx>
#include <vigra/random.hxx>
#include <iostream>
#include <limits>

namespace vigra {

template <class FeatureType>
class Signature
{
public:
    typedef typename ArrayVector<FeatureType>::const_iterator
        ConstFeatureIterator;
    typedef ArrayVector<double>::const_iterator ConstWeightIterator;

    Signature() : features_(), weights_() {}

    /** Initialize a Signature and allocate space for n bins.
    */
    Signature(int n) : features_(n), weights_(n) {}

    /** Initialize a signature by copying data from the given arrays
    */
    Signature(FeatureType *features, double* weights, int size)
        : features_(features, features + size),
        weights_(weights, weights + size) {}

    /* Return the number of bins in this signature.
    */
    int size() const
    {
        return features_.size();
    }

    /** Scale weights by the given factor.
    */
    void scale(double factor)
    {
        for (int i=0; i < weights_.size(); ++i)
        {
            weights_[i] *= factor;
        }
    }

    /** Generate a random signature with given totalWeight.
     If randomBinCount = 1, number of bins is selected randomly in
     [1, maxBin]. Otherwise, it is taken to be maxBins.
    */
    template<class Engine>
    void randomize(double totalWeight, int maxBins,
            RandomNumberGenerator<Engine> &random = RandomMT19937(),
            bool randomBinCount=1);

    ArrayVector<FeatureType> features_;
    ArrayVector<double> weights_;
};


template<typename FeatureType>
std::ostream &operator<<(std::ostream &out, const Signature<FeatureType> &sig)
{
    out.precision(15);
    out<<"signature:"<<std::endl;
    for (int i=0; i < sig.size(); ++i)
    {
        out<<"label: "<<sig.features_[i]<<" weight: "<<sig.weights_[i]<<std::endl;
    }
    return out;
}

template<> template<class Engine>
void Signature<int>::randomize(double totalWeight, int maxBins,
        RandomNumberGenerator<Engine> &random,
        bool randomBinCount)
{
    vigra_precondition(maxBins > 0,
            "Refusing to generate empty random signature.");
    // clear existing data
    features_.clear();
    weights_.clear();

    int nBins;

    if (!randomBinCount)
        nBins = maxBins;
    else
        nBins = random.uniformInt(maxBins) + 1; // avoid 0

    features_.resize(nBins);
    weights_.resize(nBins);

    double currentTotal = 0;
    for (int i=0; i < nBins; ++i)
    {
        features_[i] = i;
        weights_[i] = random.uniform(0, totalWeight*maxBins);
        currentTotal += weights_[i];
    }
    // normalize - adjust to totalWeight
    double scaleFactor = totalWeight/currentTotal;
    scale(scaleFactor);
}


class EMDFlow
{
public:
    class ArrowType
    {
    public:
        /* Feature number in signature 1 */
        int from;
        /* Feature number in signature 2 */
        int to;
        /* Amount of flow from "from" to "to" */
        double amount;

        ArrowType(int f, int t, double a) : from(f), to(t), amount(a) {};
    };

    /** Insert a new flow item.
    */
    void push_back(int from, int to, double amount)
    {
        data_.push_back(ArrowType(from, to, amount));
    }

    /** Returns the number of flow items stored.
    */
    int size() const
    {
        return data_.size();
    }

    /** Readonly access to the flow item with index i.
    */
    const ArrowType& operator[](size_t i) const
    {
        return data_[i];
    }

    /** Clear the contents of this flow.
    */
    void clear()
    {
        data_.clear();
    }

protected:

    ArrayVector<ArrowType> data_;
};

std::ostream &operator<<(std::ostream &out, const EMDFlow &flow)
{
    out<<"flow: "<<std::endl;
    out<<"from\tto\tamount"<<std::endl;
    for (int i=0; i < flow.size(); i++)
        if (flow[i].amount > 0)
            out<<flow[i].from<<" "<<flow[i].to<<" "<<flow[i].amount<<std::endl;
    return out;
}

/** \brief Set Earth Movers Distance (EMD) options.

  EMDOptions objects are used to pass options to Earth Movers Distance (EMD)
  computation functions.
*/
class EMDOptions
{
public:
    /** \brief Maximum size of bins in a signature. */
    int maxSigSize;
    /** \brief Maximum number of iterations used in the algorithm. */
    int maxIterations;
    /** \brief Error tolerance used when comparing floating point numbers. */
    double epsilon;

    /** Initialize with default values:

        - maxSigSize = 100
        - maxIterations = 500
        - epsilon = as defined by std::numeric_limits<double>::epsilon()
    */
    EMDOptions()
    : maxSigSize(100), maxIterations(500),
      epsilon(std::numeric_limits<double>::epsilon()*1e1) {}

    /** Set maximum signature size accepted by the algorithm.
    */
    EMDOptions & setMaxSigSize(int newSize)
    {
        vigra_precondition(newSize > 0,
                "Maximum signature size must be positive");
        maxSigSize = newSize;
        return *this;
    }

    /** Set maximum number of iterations.
    */
    EMDOptions & setMaxIterations(int newSize)
    {
        vigra_precondition(newSize > 0,
                "Maximum number of iterations must be positive");
        maxIterations = newSize;
        return *this;
    }

    /** Set error tolerance used to compare floating point numbers.
    */
    EMDOptions & setEpsilon(double newEpsilon)
    {
        epsilon = newEpsilon;
        return *this;
    }
};

/********************************************************/

/** Base class for different norm implemenations.

    <b> Traits defined:</b>

    <tt>FunctorTraits::isBinaryFunctor</tt> are true (<tt>VigraTrueType</tt>)
*/
template <class ValueType>
class NormBase
{
  public:
    typedef ValueType first_argument_type;
    typedef ValueType second_argument_type;
    typedef double result_type;

    double operator()(ValueType const & v1, ValueType const & v2) const;
};

template <class T>
class FunctorTraits<NormBase<T> >
: public FunctorTraitsBase<NormBase<T> >
{
public:
    typedef VigraTrueType isBinaryFunctor;
};

/** \brief L1Norm implementation.
*/
template <class ValueType>
class L1Norm : public NormBase<ValueType>
{
public:

    double operator()(ValueType const & v1, ValueType const & v2) const;
};

/** Specialize L1Norm for integers
*/
// FIXME: generalize this to arithmetic types
template <>
double L1Norm<int>::operator()(int const &v1, int const &v2) const
{
    return std::abs(v1 - v2);
}

/** \brief L2Norm implementation.
*/
template <class ValueType>
class L2Norm : public NormBase<ValueType>
{
public:

    double operator()(ValueType const & v1, ValueType const & v2) const;
};

/** Norm defined by a cost matrix.
*/
template <typename T>
class MatrixNorm;

// This only makes sense for integer values, so we implement only
// a specialization for ValueType = int.
template <>
class MatrixNorm<int> : public NormBase<int>
{
public:

    MatrixNorm(double* data, int nRows, int nCols)
        : distMatrix_(nRows, nCols, data) {}

    double operator()(int const & v1, int const & v2) const
    {
        return distMatrix_(v1, v2);
    }

protected:
    Matrix<double> distMatrix_;
};
/*****************************************************************************/

    /** \brief Compute the earth mover distance between two histograms or signatures.
    
        ADD MORE DOCUMENTATION HERE.
        
        <b> Usage:</b>

        <b>\#include</b> \<vigra/emd.hxx\><br>
        Namespace: vigra
        
        \code
        typedef ... FeatureType;
        Signature<FeatureType> s1, s2;
        ... // fill the signatures with your data
        
            // to compute the distance, pass in the signatures and a functor 
            // that returns ground distances between feature pairs
        double distance = earthMoverDistance(s1, s2, MyGroundDistance());
        
            // optionally, you can also compute the flow between the signatures
        EMDFlow flow;
        double distance = earthMoverDistance(s1, s2, MyGroundDistance(), flow);
        
            // if no ground distance functor is given, the algorithm will use
            // the default ground distance for the given FeatureType
        double distance = earthMoverDistance(s1, s2);
        
            // options can be passed by the associated option object
        double distance = earthMoverDistance(s1, s2,
                                             EMDOptions().setSomeOption());
    */
template<class FeatureType, class GroundDistanceFunctor>
double 
earthMoverDistance(Signature<FeatureType> const & signature1, 
                   Signature<FeatureType> const & signature2, 
                   GroundDistanceFunctor const & groundDistance,
                   EMDFlow & flow,
                   EMDOptions const & options = EMDOptions())
{
}

    // don't compute the flow here
template<class FeatureType, class GroundDistanceFunctor>
void 
earthMoverDistance(Signature<FeatureType> const & signature1, 
                   Signature<FeatureType> const & signature2, 
                   GroundDistanceFunctor const & groundDistance,
                   EMDOptions const & options = EMDOptions())
{
}

    // use the default ground distance for the given FeatureType
    // (the default should be deduced automatically by template matching)
template<class FeatureType>
void 
earthMoverDistance(Signature<FeatureType> const & signature1, 
                   Signature<FeatureType> const & signature2,
                   EMDFlow & flow,
                   EMDOptions const & options = EMDOptions())
{
}

    // likewise, but without computing the flow
template<class FeatureType>
void 
earthMoverDistance(Signature<FeatureType> const & signature1, 
                   Signature<FeatureType> const & signature2,
                   EMDOptions const & options = EMDOptions())
{
}

template<typename feature_t>
class EMDComputerRubner {
public:

    EMDComputerRubner() : options(EMDOptions())
    {
        initializeMemory();
    }

    EMDComputerRubner(const EMDOptions& options) : options(options)
    {
        initializeMemory();
    }

    template<template<typename ValueType> class DistanceFunctor>
    double operator()(const Signature<feature_t> &signature1,
            const Signature<feature_t> &signature2,
            DistanceFunctor<feature_t> const & func,
            EMDFlow &Flow);

    template<template<typename ValueType> class DistanceFunctor>
    double operator()(const Signature<feature_t> &signature1,
            const Signature<feature_t> &signature2,
            DistanceFunctor<feature_t> const & func);

    ~EMDComputerRubner()
    {
        for (int i=0; i < options.maxSigSize + 1; ++i)
            delete [] _C[i];
        delete [] _C;
        delete [] _X;
        for (int i=0; i < options.maxSigSize + 1; ++i)
            delete [] _IsX[i];
        delete [] _IsX;
        delete [] _RowsX;
        delete [] _ColsX;
    }
protected:
    /* NEW TYPES DEFINITION */

    /* node1_t IS USED FOR SINGLE-LINKED LISTS */
    typedef struct node1_t {
        int i;
        double val;
        struct node1_t *Next;
    } node1_t;

    /* node1_t IS USED FOR DOUBLE-LINKED LISTS */
    typedef struct node2_t {
        int i, j;
        double val;
        struct node2_t *NextC;               /* NEXT COLUMN */
        struct node2_t *NextR;               /* NEXT ROW */
    } node2_t;

    /* DECLARATION OF FUNCTIONS */
    template<template<typename ValueType> class DistanceFunctor>
    double init(const Signature<feature_t> &signature1,
            const Signature<feature_t> &signature2,
            DistanceFunctor<feature_t> const & Dist);
    void findBasicVariables(node1_t *U, node1_t *V);
    int isOptimal(node1_t *U, node1_t *V);
    int findLoop(node2_t **Loop);
    void newSol();
    void russel(double *S, double *D);
    void addBasicVariable(int minI, int minJ, double *S, double *D,
            node1_t *PrevUMinI, node1_t *PrevVMinJ,
            node1_t *UHead);
    void getFlow(const Signature<feature_t> &s1,
            const Signature<feature_t> &s2, EMDFlow &Flow);

    void initializeMemory();
    int _n1, _n2;                          /* SIGNATURES SIZES */
    double **_C;                           /* THE COST MATRIX */
    node2_t *_X;                /* THE BASIC VARIABLES VECTOR */

    /* VARIABLES TO HANDLE _X EFFICIENTLY */
    node2_t *_EndX, *_EnterX;
    char **_IsX;
    node2_t **_RowsX, **_ColsX;
    double _maxW;
    double _maxC;

    const EMDOptions &options;
};

#define EMD_DEBUG_LEVEL 0
/*
 EMD_DEBUG_LEVEL:
   0 = NO MESSAGES
   1 = PRINT THE NUMBER OF ITERATIONS AND THE FINAL RESULT
   2 = PRINT THE RESULT AFTER EVERY ITERATION
   3 = PRINT ALSO THE FLOW AFTER EVERY ITERATION
   4 = PRINT A LOT OF INFORMATION (PROBABLY USEFUL ONLY FOR THE AUTHOR)
*/

template<typename feature_t>
void EMDComputerRubner<feature_t>::getFlow(
        const Signature<feature_t> &signature1,
        const Signature<feature_t> &signature2, EMDFlow &Flow)
{
    node2_t *XP;
    for(XP=_X; XP < _EndX; XP++)
    {
        if (XP == _EnterX)  /* _EnterX IS THE EMPTY SLOT */
            continue;
        if ( (XP->i == signature1.size()) || (XP->j == signature2.size()))  /* DUMMY FEATURE */
            continue;

        if (XP->val == 0)  /* ZERO FLOW */
            continue;

        Flow.push_back(XP->i, XP->j, XP->val);
    }
}

template<typename feature_t>
template<template<typename ValueType> class DistanceFunctor>
double EMDComputerRubner<feature_t>::operator()(
        const Signature<feature_t> &signature1,
        const Signature<feature_t> &signature2,
	  DistanceFunctor<feature_t> const & Dist, EMDFlow &flow)
{
    double e = (*this)(signature1, signature2, Dist);
    getFlow(signature1, signature2, flow);
    return e;
}

template<typename feature_t>
template<template<typename ValueType> class DistanceFunctor>
double EMDComputerRubner<feature_t>::operator()(
        const Signature<feature_t> &signature1,
        const Signature<feature_t> &signature2,
	  DistanceFunctor<feature_t> const & Dist)
{
  int itr;
  double totalCost;
  double w;
  node2_t *XP;
  node1_t U[options.maxSigSize + 1], V[options.maxSigSize + 1];

  vigra_precondition(signature1.size() > 0,
          "Source signature cannot be empty!");
  vigra_precondition(signature2.size() > 0,
          "Target signature cannot be empty!");
  if (signature1.size() > options.maxSigSize ||
          signature2.size() > options.maxSigSize)
    {
        std::ostringstream s;
        s<<"emd: Signature size is limited to "<<options.maxSigSize<<std::endl;
        vigra_precondition(false, s.str());
    }

  w = init(signature1, signature2, Dist);

#if EMD_DEBUG_LEVEL > 1
  printf("\nINITIAL SOLUTION:\n");
  printSolution();
#endif

  if (_n1 > 1 && _n2 > 1)  /* IF _n1 = 1 OR _n2 = 1 THEN WE ARE DONE */
    {
      for (itr = 1; itr < options.maxIterations; itr++)
	{
	  /* FIND BASIC VARIABLES */
	  findBasicVariables(U, V);

	  /* CHECK FOR OPTIMALITY */
	  if (isOptimal(U, V))
	    break;

	  /* IMPROVE SOLUTION */
	  newSol();

#if EMD_DEBUG_LEVEL > 1
	  printf("\nITERATION # %d \n", itr);
	  printSolution();
#endif
	}

      if (itr == options.maxIterations)
          std::cerr<<"Maximum number of iterations has been reached ("<<
              options.maxIterations<<")"<<std::endl;
    }

  /* COMPUTE THE TOTAL FLOW */
  totalCost = 0;
  for(XP=_X; XP < _EndX; XP++)
    {
      if (XP == _EnterX)  /* _EnterX IS THE EMPTY SLOT */
	continue;
      if (XP->i == signature1.size() || XP->j == signature2.size())  /* DUMMY FEATURE */
	continue;

      if (XP->val == 0)  /* ZERO FLOW */
	continue;

      totalCost += (double)XP->val * _C[XP->i][XP->j];
    }

#if EMD_DEBUG_LEVEL > 0
  printf("\n*** OPTIMAL SOLUTION (%d ITERATIONS): %f ***\n", itr, totalCost);
#endif

  /* RETURN THE NORMALIZED COST == EMD */
  return (double)(totalCost / w);
}

/**********************
   init
**********************/
template<typename feature_t>
template<template<typename ValueType> class DistanceFunctor>
double EMDComputerRubner<feature_t>::init(
        const Signature<feature_t> &signature1,
        const Signature<feature_t> &signature2,
	    DistanceFunctor<feature_t> const & Dist)
{
  int i, j;
  double sSum, dSum, diff;
  double S[options.maxSigSize + 1], D[options.maxSigSize + 1];

  _n1 = signature1.size();
  _n2 = signature2.size();

  /* COMPUTE THE DISTANCE MATRIX */
  _maxC = 0;
  typename Signature<feature_t>::ConstFeatureIterator P1 =
      signature1.features_.begin();
  for(i = 0; i < _n1; i++, P1++)
  {
    typename Signature<feature_t>::ConstFeatureIterator P2 =
        signature2.features_.begin();
    for(j=0; j < _n2; j++, P2++)
      {
	_C[i][j] = Dist(*P1, *P2);
	if (_C[i][j] > _maxC)
	  _maxC = _C[i][j];
      }
  }

  /* SUM UP THE SUPPLY AND DEMAND */
  sSum = 0.0;
  for(i=0; i < _n1; i++)
    {
      S[i] = signature1.weights_[i];
      sSum += signature1.weights_[i];
      _RowsX[i] = NULL;
    }
  vigra_precondition(sSum > 0, "Total weight of source signature cannot be 0!");
  dSum = 0.0;
  for(j=0; j < _n2; j++)
    {
      D[j] = signature2.weights_[j];
      dSum += signature2.weights_[j];
      _ColsX[j] = NULL;
    }
  vigra_precondition(dSum > 0, "Total weight of target signature cannot be 0!");

  /* IF SUPPLY DIFFERENT THAN THE DEMAND, ADD A ZERO-COST DUMMY CLUSTER */
  diff = sSum - dSum;
  if (fabs(diff) >= options.epsilon * sSum)
    {
      if (diff < 0.0)
	{
	  for (j=0; j < _n2; j++)
	    _C[_n1][j] = 0;
	  S[_n1] = -diff;
	  _RowsX[_n1] = NULL;
	  _n1++;
	}
      else
	{
	  for (i=0; i < _n1; i++)
	    _C[i][_n2] = 0;
	  D[_n2] = diff;
	  _ColsX[_n2] = NULL;
	  _n2++;
	}
    }

  /* INITIALIZE THE BASIC VARIABLE STRUCTURES */
  for (i=0; i < _n1; i++)
    for (j=0; j < _n2; j++)
	_IsX[i][j] = 0;
  _EndX = _X;

  _maxW = sSum > dSum ? sSum : dSum;

  /* FIND INITIAL SOLUTION */
  russel(S, D);

  _EnterX = _EndX++;  /* AN EMPTY SLOT (ONLY _n1+_n2-1 BASIC VARIABLES) */

  return sSum > dSum ? dSum : sSum;
}

template<typename feature_t>
void EMDComputerRubner<feature_t>::initializeMemory()
{
    // use maxSigSize +1 as the size to allow for the dummy bin
    int maxSizePlusOne = options.maxSigSize + 1;
    /* THE COST MATRIX */
    _C = new double*[maxSizePlusOne];
    for (int i=0; i < maxSizePlusOne; ++i)
        _C[i] = new double[maxSizePlusOne];
    /* THE BASIC VARIABLES VECTOR */
    _X = new node2_t[maxSizePlusOne*2];

    _IsX = new char*[maxSizePlusOne];
    for (int i=0; i < maxSizePlusOne; ++i)
        _IsX[i] = new char[maxSizePlusOne];
    _RowsX = new node2_t*[maxSizePlusOne];
    _ColsX = new node2_t*[maxSizePlusOne];
}

/**********************
    findBasicVariables
 **********************/
template<typename feature_t>
void EMDComputerRubner<feature_t>::findBasicVariables(node1_t *U, node1_t *V)
{
  int i, j, found;
  int UfoundNum, VfoundNum;
  node1_t u0Head, u1Head, *CurU, *PrevU;
  node1_t v0Head, v1Head, *CurV, *PrevV;

  /* INITIALIZE THE ROWS LIST (U) AND THE COLUMNS LIST (V) */
  u0Head.Next = CurU = U;
  for (i=0; i < _n1; i++)
    {
      CurU->i = i;
      CurU->Next = CurU+1;
      CurU++;
    }
  (--CurU)->Next = NULL;
  u1Head.Next = NULL;

  CurV = V+1;
  v0Head.Next = _n2 > 1 ? V+1 : NULL;
  for (j=1; j < _n2; j++)
    {
      CurV->i = j;
      CurV->Next = CurV+1;
      CurV++;
    }
  (--CurV)->Next = NULL;
  v1Head.Next = NULL;

  /* THERE ARE _n1+_n2 VARIABLES BUT ONLY _n1+_n2-1 INDEPENDENT EQUATIONS,
     SO SET V[0]=0 */
  V[0].i = 0;
  V[0].val = 0;
  v1Head.Next = V;
  v1Head.Next->Next = NULL;

  /* LOOP UNTIL ALL VARIABLES ARE FOUND */
  UfoundNum=VfoundNum=0;
  while (UfoundNum < _n1 || VfoundNum < _n2)
    {

#if EMD_DEBUG_LEVEL > 3
      printf("UfoundNum=%d/%d,VfoundNum=%d/%d\n",UfoundNum,_n1,VfoundNum,_n2);
      printf("U0=");
      for(CurU = u0Head.Next; CurU != NULL; CurU = CurU->Next)
	printf("[%d]",CurU-U);
      printf("\n");
      printf("U1=");
      for(CurU = u1Head.Next; CurU != NULL; CurU = CurU->Next)
	printf("[%d]",CurU-U);
      printf("\n");
      printf("V0=");
      for(CurV = v0Head.Next; CurV != NULL; CurV = CurV->Next)
	printf("[%d]",CurV-V);
      printf("\n");
      printf("V1=");
      for(CurV = v1Head.Next; CurV != NULL; CurV = CurV->Next)
	printf("[%d]",CurV-V);
      printf("\n\n");
#endif

      found = 0;
      if (VfoundNum < _n2)
	{
	  /* LOOP OVER ALL MARKED COLUMNS */
	  PrevV = &v1Head;
	  for (CurV=v1Head.Next; CurV != NULL; CurV=CurV->Next)
	    {
	      j = CurV->i;
	      /* FIND THE VARIABLES IN COLUMN j */
	      PrevU = &u0Head;
	      for (CurU=u0Head.Next; CurU != NULL; CurU=CurU->Next)
		{
		  i = CurU->i;
		  if (_IsX[i][j])
		    {
		      /* COMPUTE U[i] */
		      CurU->val = _C[i][j] - CurV->val;
		      /* ...AND ADD IT TO THE MARKED LIST */
		      PrevU->Next = CurU->Next;
		      CurU->Next = u1Head.Next != NULL ? u1Head.Next : NULL;
		      u1Head.Next = CurU;
		      CurU = PrevU;
		    }
		  else
		    PrevU = CurU;
		}
	      PrevV->Next = CurV->Next;
	      VfoundNum++;
	      found = 1;
	    }
	}
     if (UfoundNum < _n1)
	{
	  /* LOOP OVER ALL MARKED ROWS */
	  PrevU = &u1Head;
	  for (CurU=u1Head.Next; CurU != NULL; CurU=CurU->Next)
	    {
	      i = CurU->i;
	      /* FIND THE VARIABLES IN ROWS i */
	      PrevV = &v0Head;
	      for (CurV=v0Head.Next; CurV != NULL; CurV=CurV->Next)
		{
		  j = CurV->i;
		  if (_IsX[i][j])
		    {
		      /* COMPUTE V[j] */
		      CurV->val = _C[i][j] - CurU->val;
		      /* ...AND ADD IT TO THE MARKED LIST */
		      PrevV->Next = CurV->Next;
		      CurV->Next = v1Head.Next != NULL ? v1Head.Next: NULL;
		      v1Head.Next = CurV;
		      CurV = PrevV;
		    }
		  else
		    PrevV = CurV;
		}
	      PrevU->Next = CurU->Next;
	      UfoundNum++;
	      found = 1;
	    }
	}
     if (! found)
     {
         vigra_fail("emd: Unexpected error in findBasicVariables!\n"
                 "This typically happens when epsilon defined in\n"
                 "EMDOptions not right for the scale of the problem.");
     }
    }
}

/**********************
    isOptimal
 **********************/
template<typename feature_t>
int EMDComputerRubner<feature_t>::isOptimal(node1_t *U, node1_t *V)
{
  double delta, deltaMin;
  int i, j, minI = 0, minJ = 0;

  /* FIND THE MINIMAL Cij-Ui-Vj OVER ALL i,j */
  deltaMin = INFINITY;
  for(i=0; i < _n1; i++)
    for(j=0; j < _n2; j++)
      if (! _IsX[i][j])
	{
	  delta = _C[i][j] - U[i].val - V[j].val;
	  if (deltaMin > delta)
	    {
              deltaMin = delta;
	      minI = i;
	      minJ = j;
	    }
	}

#if EMD_DEBUG_LEVEL > 3
  printf("deltaMin=%f\n", deltaMin);
#endif

   if (std::isinf(deltaMin))
     {
       vigra_fail("emd: Unexpected error in isOptimal.");
     }

   _EnterX->i = minI;
   _EnterX->j = minJ;

   /* IF NO NEGATIVE deltaMin, WE FOUND THE OPTIMAL SOLUTION */
   return deltaMin >= -options.epsilon * _maxC;

/*
   return deltaMin >= -options.epsilon;
 */
}

/**********************
    newSol
**********************/
template<typename feature_t>
void EMDComputerRubner<feature_t>::newSol()
{
    int i, j, k;
    double xMin;
    int steps;
    node2_t *Loop[2*options.maxSigSize + 1], *CurX, *LeaveX = NULL;

#if EMD_DEBUG_LEVEL > 3
    printf("EnterX = (%d,%d)\n", _EnterX->i, _EnterX->j);
#endif

    /* ENTER THE NEW BASIC VARIABLE */
    i = _EnterX->i;
    j = _EnterX->j;
    _IsX[i][j] = 1;
    _EnterX->NextC = _RowsX[i];
    _EnterX->NextR = _ColsX[j];
    _EnterX->val = 0;
    _RowsX[i] = _EnterX;
    _ColsX[j] = _EnterX;

    /* FIND A CHAIN REACTION */
    steps = findLoop(Loop);

    /* FIND THE LARGEST VALUE IN THE LOOP */
    xMin = INFINITY;
    for (k=1; k < steps; k+=2)
      {
	if (Loop[k]->val < xMin)
	  {
	    LeaveX = Loop[k];
	    xMin = Loop[k]->val;
	  }
      }

    /* UPDATE THE LOOP */
    for (k=0; k < steps; k+=2)
      {
	Loop[k]->val += xMin;
	Loop[k+1]->val -= xMin;
      }

#if EMD_DEBUG_LEVEL > 3
    printf("LeaveX = (%d,%d)\n", LeaveX->i, LeaveX->j);
#endif

    /* REMOVE THE LEAVING BASIC VARIABLE */
    i = LeaveX->i;
    j = LeaveX->j;
    _IsX[i][j] = 0;
    if (_RowsX[i] == LeaveX)
      _RowsX[i] = LeaveX->NextC;
    else
      for (CurX=_RowsX[i]; CurX != NULL; CurX = CurX->NextC)
	if (CurX->NextC == LeaveX)
	  {
	    CurX->NextC = CurX->NextC->NextC;
	    break;
	  }
    if (_ColsX[j] == LeaveX)
      _ColsX[j] = LeaveX->NextR;
    else
      for (CurX=_ColsX[j]; CurX != NULL; CurX = CurX->NextR)
	if (CurX->NextR == LeaveX)
	  {
	    CurX->NextR = CurX->NextR->NextR;
	    break;
	  }

    /* SET _EnterX TO BE THE NEW EMPTY SLOT */
    _EnterX = LeaveX;
}

/**********************
    findLoop
**********************/
template<typename feature_t>
int EMDComputerRubner<feature_t>::findLoop(node2_t **Loop)
{
  int i, steps;
  node2_t **CurX, *NewX;
  char IsUsed[2*options.maxSigSize + 1];

  for (i=0; i < _n1+_n2; i++)
    IsUsed[i] = 0;

  CurX = Loop;
  NewX = *CurX = _EnterX;
  IsUsed[_EnterX-_X] = 1;
  steps = 1;

  do
    {
      if (steps%2 == 1)
	{
	  /* FIND AN UNUSED X IN THE ROW */
	  NewX = _RowsX[NewX->i];
	  while (NewX != NULL && IsUsed[NewX-_X])
	    NewX = NewX->NextC;
	}
      else
	{
	  /* FIND AN UNUSED X IN THE COLUMN, OR THE ENTERING X */
	  NewX = _ColsX[NewX->j];
	  while (NewX != NULL && IsUsed[NewX-_X] && NewX != _EnterX)
	    NewX = NewX->NextR;
	  if (NewX == _EnterX)
	    break;
	}

     if (NewX != NULL)  /* FOUND THE NEXT X */
       {
	 /* ADD X TO THE LOOP */
	 *++CurX = NewX;
	 IsUsed[NewX-_X] = 1;
	 steps++;
#if EMD_DEBUG_LEVEL > 3
	 printf("steps=%d, NewX=(%d,%d)\n", steps, NewX->i, NewX->j);
#endif
       }
     else  /* DIDN'T FIND THE NEXT X */
       {
	 /* BACKTRACK */
	 do
	   {
	     NewX = *CurX;
	     do
	       {
		 if (steps%2 == 1)
		   NewX = NewX->NextR;
		 else
		   NewX = NewX->NextC;
	       } while (NewX != NULL && IsUsed[NewX-_X]);

	     if (NewX == NULL)
	       {
		 IsUsed[*CurX-_X] = 0;
		 CurX--;
		 steps--;
	       }
	 } while (NewX == NULL && CurX >= Loop);

#if EMD_DEBUG_LEVEL > 3
	 printf("BACKTRACKING TO: steps=%d, NewX=(%d,%d)\n",
		steps, NewX->i, NewX->j);
#endif
           IsUsed[*CurX-_X] = 0;
	   *CurX = NewX;
	   IsUsed[NewX-_X] = 1;
       }
    } while(CurX >= Loop);

  if (CurX == Loop)
    {
      vigra_fail("emd: Unexpected error in findLoop!");
    }
#if EMD_DEBUG_LEVEL > 3
  printf("FOUND LOOP:\n");
  for (i=0; i < steps; i++)
    printf("%d: (%d,%d)\n", i, Loop[i]->i, Loop[i]->j);
#endif

  return steps;
}

/**********************
    russel
**********************/
template<typename feature_t>
void EMDComputerRubner<feature_t>::russel(double *S, double *D)
{
  int i, j, found, minI = 0, minJ = 0;
  double deltaMin, oldVal, diff;
  double Delta[options.maxSigSize + 1][options.maxSigSize + 1];
  node1_t Ur[options.maxSigSize + 1], Vr[options.maxSigSize + 1];
  node1_t uHead, *CurU, *PrevU;
  node1_t vHead, *CurV, *PrevV;
  node1_t *PrevUMinI = NULL, *PrevVMinJ = NULL, *Remember;

  /* INITIALIZE THE ROWS LIST (Ur), AND THE COLUMNS LIST (Vr) */
  uHead.Next = CurU = Ur;
  for (i=0; i < _n1; i++)
    {
      CurU->i = i;
      CurU->val = -INFINITY;
      CurU->Next = CurU+1;
      CurU++;
    }
  (--CurU)->Next = NULL;

  vHead.Next = CurV = Vr;
  for (j=0; j < _n2; j++)
    {
      CurV->i = j;
      CurV->val = -INFINITY;
      CurV->Next = CurV+1;
      CurV++;
    }
  (--CurV)->Next = NULL;

  /* FIND THE MAXIMUM ROW AND COLUMN VALUES (Ur[i] AND Vr[j]) */
  for(i=0; i < _n1 ; i++)
    for(j=0; j < _n2 ; j++)
      {
	double v;
	v = _C[i][j];
	if (Ur[i].val <= v)
	  Ur[i].val = v;
	if (Vr[j].val <= v)
	  Vr[j].val = v;
      }

  /* COMPUTE THE Delta MATRIX */
  for(i=0; i < _n1 ; i++)
    for(j=0; j < _n2 ; j++)
      Delta[i][j] = _C[i][j] - Ur[i].val - Vr[j].val;

  /* FIND THE BASIC VARIABLES */
  do
    {
#if EMD_DEBUG_LEVEL > 3
      printf("Ur=");
      for(CurU = uHead.Next; CurU != NULL; CurU = CurU->Next)
	printf("[%d]",CurU-Ur);
      printf("\n");
      printf("Vr=");
      for(CurV = vHead.Next; CurV != NULL; CurV = CurV->Next)
	printf("[%d]",CurV-Vr);
      printf("\n");
      printf("\n\n");
#endif

      /* FIND THE SMALLEST Delta[i][j] */
      found = 0;
      deltaMin = INFINITY;
      PrevU = &uHead;
      for (CurU=uHead.Next; CurU != NULL; CurU=CurU->Next)
	{
	  int i;
	  i = CurU->i;
	  PrevV = &vHead;
	  for (CurV=vHead.Next; CurV != NULL; CurV=CurV->Next)
	    {
	      int j;
	      j = CurV->i;
	      if (deltaMin > Delta[i][j])
		{
		  deltaMin = Delta[i][j];
		  minI = i;
		  minJ = j;
		  PrevUMinI = PrevU;
		  PrevVMinJ = PrevV;
		  found = 1;
		}
	      PrevV = CurV;
	    }
	  PrevU = CurU;
	}

      if (! found)
	break;

      /* ADD X[minI][minJ] TO THE BASIS, AND ADJUST SUPPLIES AND COST */
      Remember = PrevUMinI->Next;
      addBasicVariable(minI, minJ, S, D, PrevUMinI, PrevVMinJ, &uHead);

      /* UPDATE THE NECESSARY Delta[][] */
      if (Remember == PrevUMinI->Next)  /* LINE minI WAS DELETED */
	{
	  for (CurV=vHead.Next; CurV != NULL; CurV=CurV->Next)
	    {
	      int j;
	      j = CurV->i;
	      if (CurV->val == _C[minI][j])  /* COLUMN j NEEDS UPDATING */
		{
		  /* FIND THE NEW MAXIMUM VALUE IN THE COLUMN */
		  oldVal = CurV->val;
		  CurV->val = -INFINITY;
		  for (CurU=uHead.Next; CurU != NULL; CurU=CurU->Next)
		    {
		      int i;
		      i = CurU->i;
		      if (CurV->val <= _C[i][j])
			CurV->val = _C[i][j];
		    }

		  /* IF NEEDED, ADJUST THE RELEVANT Delta[*][j] */
		  diff = oldVal - CurV->val;
		  if (fabs(diff) < options.epsilon * _maxC)
		    for (CurU=uHead.Next; CurU != NULL; CurU=CurU->Next)
		      Delta[CurU->i][j] += diff;
		}
	    }
	}
      else  /* COLUMN minJ WAS DELETED */
	{
	  for (CurU=uHead.Next; CurU != NULL; CurU=CurU->Next)
	    {
	      int i;
	      i = CurU->i;
	      if (CurU->val == _C[i][minJ])  /* ROW i NEEDS UPDATING */
		{
		  /* FIND THE NEW MAXIMUM VALUE IN THE ROW */
		  oldVal = CurU->val;
		  CurU->val = -INFINITY;
		  for (CurV=vHead.Next; CurV != NULL; CurV=CurV->Next)
		    {
		      int j;
		      j = CurV->i;
		      if(CurU->val <= _C[i][j])
			CurU->val = _C[i][j];
		    }

		  /* If NEEDED, ADJUST THE RELEVANT Delta[i][*] */
		  diff = oldVal - CurU->val;
		  if (fabs(diff) < options.epsilon * _maxC)
		    for (CurV=vHead.Next; CurV != NULL; CurV=CurV->Next)
		      Delta[i][CurV->i] += diff;
		}
	    }
	}
    } while (uHead.Next != NULL || vHead.Next != NULL);
}

/**********************
    addBasicVariable
**********************/
template<typename feature_t>
void EMDComputerRubner<feature_t>::addBasicVariable(int minI, int minJ,
        double *S, double *D,
			     node1_t *PrevUMinI, node1_t *PrevVMinJ,
			     node1_t *UHead)
{
  double T;

  if (fabs(S[minI]-D[minJ]) <= options.epsilon * _maxW)  /* DEGENERATE CASE */
    {
      T = S[minI];
      S[minI] = 0;
      D[minJ] -= T;
    }
  else if (S[minI] < D[minJ])  /* SUPPLY EXHAUSTED */
    {
      T = S[minI];
      S[minI] = 0;
      D[minJ] -= T;
    }
  else  /* DEMAND EXHAUSTED */
    {
      T = D[minJ];
      D[minJ] = 0;
      S[minI] -= T;
    }

  /* X(minI,minJ) IS A BASIC VARIABLE */
  _IsX[minI][minJ] = 1;

  _EndX->val = T;
  _EndX->i = minI;
  _EndX->j = minJ;
  _EndX->NextC = _RowsX[minI];
  _EndX->NextR = _ColsX[minJ];
  _RowsX[minI] = _EndX;
  _ColsX[minJ] = _EndX;
  _EndX++;

  /* DELETE SUPPLY ROW ONLY IF THE EMPTY, AND IF NOT LAST ROW */
  if (S[minI] == 0 && UHead->Next->Next != NULL)
    PrevUMinI->Next = PrevUMinI->Next->Next;  /* REMOVE ROW FROM LIST */
  else
    PrevVMinJ->Next = PrevVMinJ->Next->Next;  /* REMOVE COLUMN FROM LIST */
}

} // namespace vigra

#endif // VIGRA_EMD_HXX

