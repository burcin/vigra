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
/************************************************************************/

#ifndef VIGRA_EMD_HXX
#define VIGRA_EMD_HXX

#include <vigra/error.hxx>
#include <vigra/array_vector.hxx>
#include <vigra/random.hxx>
#include <ostream>
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
    // implement this class
    // (used to return the flow between the histograms, see the paper)
};

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

// implement suitable ground distance functors for the most common feature types

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

}

#endif // VIGRA_EMD_HXX

